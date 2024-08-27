import awkward
import numpy

from correctionlib import _core

from higgs_dna.utils import misc_utils, awkward_utils

###################################
### b-tag continuous reshape SF ###
###################################

BTAG_RESHAPE_SF_FILE = {
    "2016" : "jsonpog-integration/POG/BTV/2016postVFP_UL/btagging.json", 
    "2016UL_preVFP" : "jsonpog-integration/POG/BTV/2016preVFP_UL/btagging.json", 
    "2016UL_postVFP" : "jsonpog-integration/POG/BTV/2016postVFP_UL/btagging.json", 
    "2017" : "jsonpog-integration/POG/BTV/2017_UL/btagging.json",
    "2018" : "jsonpog-integration/POG/BTV/2018_UL/btagging.json"
}

DEEPJET_RESHAPE_SF = {
    "2016" : "deepJet_shape", 
    "2016UL_preVFP" : "deepJet_shape",
    "2016UL_postVFP" : "deepJet_shape",
    "2017" : "deepJet_shape",
    "2018" : "deepJet_shape"
}
PUJETID_SF_FILE = {
    "2016UL_preVFP" : "jsonpog-integration/POG/JME/2016preVFP_UL/jmar.json", 
    "2016UL_postVFP" : "jsonpog-integration/POG/JME/2016postVFP_UL/jmar.json", 
    "2017" : "jsonpog-integration/POG/JME/2017_UL/jmar.json",
    "2018" : "jsonpog-integration/POG/JME/2018_UL/jmar.json"
}

PUJETID_SF = {
    "2016UL_preVFP" : "PUJetID_eff",
    "2016UL_postVFP" : "PUJetID_eff",
    "2017" : "PUJetID_eff",
    "2018" : "PUJetID_eff"
}

DEEPJET_VARIATIONS = { 
    "up_jes" : [5, 0], # applicable to b (5) and light (0) jets, but not charm (4)
    "up_lf" : [5],
    "up_hfstats1" : [5],
    "up_hfstats2" : [5],
    "up_cferr1" : [4],
    "up_cferr2" : [4],
    "up_hf" : [0],
    "up_lfstats1" : [0],
    "up_lfstats2" : [0],
    "down_jes" : [5, 0], # applicable to b (5) and light (0) jets, but not charm(4)
    "down_lf" : [5],
    "down_hfstats1" : [5],
    "down_hfstats2" : [5],
    "down_cferr1" : [4],
    "down_cferr2" : [4],
    "down_hf" : [0],
    "down_lfstats1" : [0],
    "down_lfstats2" : [0],
}

def btag_deepjet_reshape_sf(events, year, central_only, input_collection):
    """
    See:
        - https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/BTV_bjets_Run2_UL/
        - https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/btvExample.py

    Note: application of SFs should not change the overall normalization of a sample (before any b-tagging selection) and each sample should be adjusted by an overall weight derived in a phase space with no requirements on b-jets such that the normalization is unchanged. TODO: link BTV TWiki that describes this.
    """
    required_fields = [
        (input_collection, "eta"), (input_collection, "pt"), (input_collection, "hadronFlavour"), (input_collection, "btagDeepFlavB") 
    ]
    missing_fields = awkward_utils.missing_fields(events, required_fields)

    evaluator = _core.CorrectionSet.from_file(misc_utils.expand_path(BTAG_RESHAPE_SF_FILE[year]))
   
    jets = events[input_collection]
    jets["flavor"] = jets.hadronFlavour

    # Flatten jets then convert to numpy for compatibility with correctionlib
    n_jets = awkward.num(jets) # save n_jets to convert back to jagged format at the end 
    jets_flattened = awkward.flatten(jets)

    jet_flavor = awkward.to_numpy(jets_flattened.flavor)
    jet_abs_eta = numpy.clip(
        awkward.to_numpy(abs(jets_flattened.eta)),
        0.0,
        2.49999 # SFs only valid up to eta 2.5
    )
    jet_pt = numpy.clip(
        awkward.to_numpy(jets_flattened.pt),
        20.0, # SFs only valid for pT > 20.
        99999999.
    )
    jet_disc = awkward.to_numpy(jets_flattened.btagDeepFlavB)        

    variations_list = ["central"]
    if not central_only:
        variations_list += DEEPJET_VARIATIONS.keys()

    variations = {}

    central_sf = evaluator[DEEPJET_RESHAPE_SF[year]].evalv(
            "central",
            jet_flavor,
            jet_abs_eta,
            jet_pt,
            jet_disc
    )

    variations["central"] = awkward.unflatten(central_sf, n_jets)

    for var in variations_list:
        if var == "central":
            continue
        applicable_flavors = DEEPJET_VARIATIONS[var] # the up/down variations are only applicable to specific flavors of jet
        var_sf = central_sf 
        for f in applicable_flavors:
            var_sf = numpy.where(
                jet_flavor == f,
                evaluator[DEEPJET_RESHAPE_SF[year]].evalv(
                    var,
                    numpy.ones_like(jet_flavor) * f,
                    jet_abs_eta,
                    jet_pt,
                    jet_disc
                ),
                var_sf
            )

        variations[var] = awkward.unflatten(var_sf, n_jets) # make jagged again

    for var in variations.keys():
        # Set SFs = 1 for jets which are not applicable (pt <= 20 or |eta| >= 2.5)
        variations[var] = awkward.where(
                (jets.pt <= 20.0) | (abs(jets.eta) >= 2.5),
                awkward.ones_like(variations[var]),
                variations[var]
        )

    return variations


def PUJetID_sf(events, year,central_only, input_collection, working_point = "none"):
    """
    See:
        -https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL
        
    Note: PUJetID SFs are applied to the jets in the event.
    """
    required_fields = [
        (input_collection, "eta"), (input_collection, "pt")
    ]

    missing_fields = awkward_utils.missing_fields(events, required_fields)

    evaluator = _core.CorrectionSet.from_file(misc_utils.expand_path(PUJETID_SF_FILE[year]))

    jets = events[input_collection]

    # Flatten jets then convert to numpy for compatibility with correctionlib
    n_jets = awkward.num(jets)
    jets_flattened = awkward.flatten(jets)

    jet_pt = numpy.clip(
        awkward.to_numpy(jets_flattened.pt),
        12.5, # SFs only valid for pT >= 20.0
        57.49999
    )

    jet_eta = awkward.to_numpy(jets_flattened.eta)
    # Calculate SF and syst
    variations = {} 
    PUJETID_SFNAME = PUJETID_SF[year]
    sf = evaluator[PUJETID_SFNAME].evalv(
            jet_eta,
            jet_pt,            
            "nom",
            working_point
    )
    variations["central"] = awkward.unflatten(sf, n_jets)
    if not central_only:
        syst_vars = ["up", "down"]
        for syst_var in syst_vars:
            syst = evaluator[PUJETID_SFNAME].evalv(
            jet_eta,
            jet_pt,            
            syst_var,
            working_point
    )
            if "up" in syst_var:
                syst_var_name = "up"
            elif "down" in syst_var:
                syst_var_name = "down"
            variations[syst_var_name] = awkward.unflatten(syst, n_jets)

    for var in variations.keys():
        # Set SFs = 1 for jets which are not applicable
        variations[var] = awkward.where(
                jets.pt > 57.49999,
                awkward.ones_like(variations[var]),
                variations[var]
        )

    return variations
########################
### Photon ID MVA SF ###
########################
Jet_ID_SF_FILE = {
    "2016" : "jsonpog-integration/POG/JME/2016postVFP_UL/jmar.json",
    "2016UL_preVFP" : "jsonpog-integration/POG/JME/2016preVFP_UL/jmar.json",
    "2016UL_postVFP" : "jsonpog-integration/POG/JME/2016postVFP_UL/jmar.json",
    "2017" : "jsonpog-integration/POG/JME/2017_UL/jmar.json",
    "2018" : "jsonpog-integration/POG/JME/2018_UL/jmar.json"
}

Jet_ID_SF = {
    "2016" : "2016postVFP",
    "2016UL_preVFP" : "2016preVFP",
    "2016UL_postVFP" : "2016postVFP",
    "2017" : "2017",
    "2018" : "2018"
}


def WvsQCD_MD_sf(events, year, central_only, input_collection, working_point = "none"):
    """
    See: 
        - https://twiki.cern.ch/twiki/bin/viewauth/CMS/ParticleNetSFs#W_Tagger_MD
    """

    required_fields = [
        (input_collection, "eta"), (input_collection, "pt")
    ]

    missing_fields = awkward_utils.missing_fields(events, required_fields)

    evaluator = _core.CorrectionSet.from_file(misc_utils.expand_path(Jet_ID_SF_FILE[year]))

    fatjets = events[input_collection]

    # Flatten fatjets then convert to numpy for compatibility with correctionlib
    n_fatjets = awkward.num(fatjets)
    fatjets_flattened = awkward.flatten(fatjets)

    fatjet_pt = numpy.clip(
        awkward.to_numpy(fatjets_flattened.pt),
        200.0, 
        799.9999
    )
    if year == "2016UL_preVFP" or year == "2016UL_postVFP":
        fatjet_eta = numpy.clip(
            awkward.to_numpy(fatjets_flattened.eta),
            -2.3999,
            2.3999
        )
    if year == "2018" or year == "2017":
        fatjet_eta = numpy.clip(
            awkward.to_numpy(fatjets_flattened.eta),
            -2.4999,
            2.4999
        )
    # Calculate SF and syst
    variations = {} 
    sf = evaluator['ParticleNet_W_MD'].evalv(
            fatjet_eta,
            fatjet_pt,
            "nom",
            working_point
    )
    variations["central"] = awkward.unflatten(sf, n_fatjets)
    if not central_only:
        syst_vars = ["up", "down"]
        for syst_var in syst_vars:
            syst = evaluator['ParticleNet_W_MD'].evalv(
                    fatjet_eta,
                    fatjet_pt,
                    syst_var,
                    working_point
            )
            if "up" in syst_var:
                syst_var_name = "up"
            elif "down" in syst_var:
                syst_var_name = "down"
            variations[syst_var_name] = awkward.unflatten(syst, n_fatjets)

    for var in variations.keys():
        # Set SFs = 1 for fatjets which are not applicable
        variations[var] = awkward.where(
                ((fatjets.pt < 200.0)|(fatjets.pt >= 800.0)),
                awkward.ones_like(variations[var]),
                variations[var]
        )
        variations[var] = awkward.where(
                ((fatjets.eta <= 2.5)|(fatjets.eta >= 2.5)),
                awkward.ones_like(variations[var]),
                variations[var]
        )

    return variations

PNbb_SF_FILE = {
    "2016" : "higgs_dna/systematics/data/PNetbb.json",
    "2016UL_preVFP" : "higgs_dna/systematics/data/PNetbb.json",
    "2016UL_postVFP" : "higgs_dna/systematics/data/PNetbb.json",
    "2017" : "higgs_dna/systematics/data/PNetbb.json",
    "2018" : "higgs_dna/systematics/data/PNetbb.json"
}

PNbb_WP = {
    "2016UL_preVFP" : "PNbbSF16pre",
    "2016UL_postVFP" : "PNbbSF16post",
    "2017" : "PNbbSF17",
    "2018" : "PNbbSF18"
}
def PNbb_veto_sf(events, year, central_only, input_collection):

    required_fields = [
        (input_collection, "pt")
    ]

    missing_fields = awkward_utils.missing_fields(events, required_fields)

    evaluator = _core.CorrectionSet.from_file(misc_utils.expand_path(PNbb_SF_FILE[year]))

    fatjets = events[input_collection]

    # Flatten fatjets then convert to numpy for compatibility with correctionlib
    n_fatjets = awkward.num(fatjets)
    fatjets_flattened = awkward.flatten(fatjets)

    fatjet_pt = numpy.clip(
        awkward.to_numpy(fatjets_flattened.pt),
        100.0, 
        2999.9999
    )
    # Calculate SF and syst
    variations = {} 
    working_point = PNbb_WP[year]
    sf = evaluator[working_point].evalv(
            fatjet_pt,
    )
    variations["central"] = awkward.unflatten(sf, n_fatjets)
    if not central_only:
        syst_vars = ["up", "down"]
        for syst_var in syst_vars:
            working_point = PNbb_WP[year]+syst_var
            syst = evaluator[working_point].evalv(
                    fatjet_pt,
            )
            if "up" in syst_var:
                syst_var_name = "up"
            elif "down" in syst_var:
                syst_var_name = "down"
            variations[syst_var_name] = awkward.unflatten(syst, n_fatjets)

    for var in variations.keys():
        # Set SFs = 1 for fatjets which are not applicable
        variations[var] = awkward.where(
                (fatjets.pt < 100.0),
                awkward.ones_like(variations[var]),
                variations[var]
        )

    return variations
 
    