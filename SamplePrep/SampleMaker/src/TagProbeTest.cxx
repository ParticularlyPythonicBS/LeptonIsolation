// local tools
#include "ObjectFilters.cxx"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "xAODEgamma/ElectronContainer.h"
#include "xAODEgamma/Electron.h"
#include "xAODMuon/MuonContainer.h"
#include "xAODMuon/Muon.h"
#include "xAODTracking/TrackParticleContainer.h"
#include "xAODTracking/TrackParticle.h"
#include "xAODEgamma/EgammaxAODHelpers.h"
#include "xAODTruth/xAODTruthHelpers.h"
#include "xAODTracking/TrackParticlexAODHelpers.h"
#include "InDetTrackSelectionTool/InDetTrackSelectionTool.h"

// AnalysisBase tool include(s):
#include "xAODRootAccess/Init.h"
#include "xAODRootAccess/TEvent.h"
#include "xAODRootAccess/tools/ReturnCheck.h"

// 3rd party includes
#include "TFile.h"
#include "H5Cpp.h"

// stl includes
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include <cassert>

using namespace std;

int main (int argc, char *argv[]) {

    // parse input
    const char* ALG = argv[0];
    string inputFilename = argv[1];

    // object filters
    ObjectFilters object_filters;

    // Open the file:
    unique_ptr<TFile> ifile(TFile::Open(inputFilename.c_str(), "READ"));
    if ( ! ifile.get() || ifile->IsZombie()) {
        throw logic_error("Couldn't open file: " + inputFilename);
        return 1;
    }
    cout << "Opened file: " << inputFilename << endl;

    // Connect the event object to it:
    RETURN_CHECK(ALG, xAOD::Init());
    xAOD::TEvent event(xAOD::TEvent::kClassAccess);
    RETURN_CHECK(ALG, event.readFrom(ifile.get()));

    // Leptons
    TFile outputFile("output.root", "recreate");
    TTree* outputTree = new TTree("BaselineTree", "baseline tree");

    int pdgID; outputTree->Branch("pdgID", &pdgID, "pdgID/I");
    int truth_type; outputTree->Branch("truth_type", &truth_type, "truth_type/I");
    int isolation; outputTree->Branch("isolation", &isolation, "isolation/I");  //added for tag and probe

    int entries = event.getEntries();
    entries = 1000;
    cout << "got " << entries << " entries" << endl;

    cout << "\nProcessing leptons" << endl;
    for (entry_n = 0; entry_n < entries; ++entry_n) {

        // Print some status
        if ( ! (entry_n % 500)) {
            cout << "Processing " << entry_n << "/" << entries << "\n";
        }

        // Load the event
        bool ok = event.getEntry(entry_n) >= 0;
        if (!ok) throw logic_error("getEntry failed");

        // Get tracks and leptons
        const xAOD::ElectronContainer *electrons;
        RETURN_CHECK(ALG, event.retrieve(electrons, "Electrons"));
        const xAOD::MuonContainer *muons;
        RETURN_CHECK(ALG, event.retrieve(muons, "Muons"));

        // Filter objects
        vector<const xAOD::Electron*> probe_filtered_electrons = object_filters.filter_electrons_probe(electrons);
        vector<const xAOD::Muon*> probe_filtered_muons = object_filters.filter_muons_probe(muons);
        vector<const xAOD::Electron*> tag_filtered_electrons = object_filters.filter_electrons_tag(electrons);
        vector<const xAOD::Muon*> tag_filtered_muons = object_filters.filter_muons_tag(muons);
        
        //check if there can be tags
        if (tight_filtered_muons.size() <1 || tight_filtered_electrons.size() < 1)
            continue;
         //Event filter
        if (baseline_filtered_electrons.size() < 2)
            continue;
        if (baseline__muons.size() < 2)
            continue;        //if len(lepton) < 2: continue
       
        vector <pair<pair<const xAOD::Electron*,int>>,int> checked_electrons;
        vector <pair<pair<const xAOD::Muon*, int>>, int> checked_muons;

        for (auto electron1 : tag_filtered_electrons){
            for (auto electron2 : probe_filtered_electrons){
                if (electron1 == electron2) continue;
                double mass = (electron1.first->p4() + electron2.first->p4()).M() 
                if (mass > 81 && mass < 101):
                {
                    if (electron1.charge()*electron2.charge()< 0):
                    {
                        checked_electrons.push_back(make_pair(electron2, 1));//add to tuple for signal
                    }
                    else
                    {
                        checked_electrons.push_back(make_pair(electron2, -1)); //add to tuple for background
                    }
                }//1 for OSSF, -1 for SSSF 
            }
        }//tag and probe for electrons

        for (auto muon1 : filtered_muons){
            for (auto muon2 : filtered_muons){
                if (muon1 == muon2) continue;
                double mass = (muon1.first->p4() + muon2.first->p4()).M()
                if (mass > 81 && mass < 120):
                {
                    if (muon1.charge()*muon2.charge()< 0):
                    {
                        checked_muons.push_back(make_pair(muon2,1));//add to tuple for signal
                    }
                    else
                    {
                        checked_muons.push_back(make_pair(muon2,-1));//add to tuple for background
                    }
                }//1 for OSSF, -1 for SSSF 
            } 
        }//tag and probe for muons
        
        // one of the electrons in the pairs: Pt >10GeV, high confidence-> tag
        // the other Pt>5GeV, opposite sign -> probe
        // background is sign flip
        
        
        // Write event
        SG::AuxElement::ConstAccessor<float> accessPromptVar("PromptLeptonVeto");

        for (auto electron : filtered_electrons) {
            truth_type = xAOD::TruthHelpers::getParticleTruthType(*electron); // 2 = real prompt, 3 = HF
            if (truth_type != 2 && truth_type != 3) continue;

            pdgID = 11;
        }

        for (auto muon : filtered_muons) {
            truth_type = xAOD::TruthHelpers::getParticleTruthType(*(muon->trackParticle(xAOD::Muon::InnerDetectorTrackParticle))); // 2 = real prompt, 3 = HF
            if (truth_type != 2 && truth_type != 3) continue;

            pdgID = 13;
        }
    }

    outputTree->Write();
    outputFile.Close();

    return 0;
}
