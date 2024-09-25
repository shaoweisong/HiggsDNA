import awkward
import numpy

from correctionlib import _core
import vector
import logging
logger = logging.getLogger(__name__)
from higgs_dna.utils import awkward_utils, misc_utils
from higgs_dna.systematics.utils import systematic_from_bins

ELECTRON_ID_SF_FILE = {
    "2016" : "jsonpog-integration/POG/EGM/2016postVFP_UL/electron.json",
    "2016UL_preVFP" : "jsonpog-integration/POG/EGM/2016preVFP_UL/electron.json",
    "2016UL_postVFP" : "jsonpog-integration/POG/EGM/2016postVFP_UL/electron.json",
    "2017" : "jsonpog-integration/POG/EGM/2017_UL/electron.json",
    "2018" : "jsonpog-integration/POG/EGM/2018_UL/electron.json"
}
MUON_ID_SF_FILE = {
    "2016" : "jsonpog-integration/POG/MUO/2016postVFP_UL/muon_Z.json",
    "2016UL_preVFP" : "jsonpog-integration/POG/MUO/2016preVFP_UL/muon_Z.json",
    "2016UL_postVFP" : "jsonpog-integration/POG/MUO/2016postVFP_UL/muon_Z.json",
    "2017" : "jsonpog-integration/POG/MUO/2017_UL/muon_Z.json",
    "2018" : "jsonpog-integration/POG/MUO/2018_UL/muon_Z.json"
}
MUON_HIGHPTID_SF_FILE = {
    "2016" : "jsonpog-integration/POG/MUO/2016postVFP_UL/muon_Z.json",
    "2016UL_preVFP" : "jsonpog-integration/POG/MUO/2016preVFP_UL/ScaleFactors_Muon_highPt_IDISO_2016_preVFP_schemaV2.json",
    "2016UL_postVFP" : "jsonpog-integration/POG/MUO/2016postVFP_UL/ScaleFactors_Muon_highPt_IDISO_2016_schemaV2.json",
    "2017" : "jsonpog-integration/POG/MUO/2017_UL/ScaleFactors_Muon_highPt_IDISO_2017_schemaV2.json",
    "2018" : "jsonpog-integration/POG/MUO/2018_UL/ScaleFactors_Muon_highPt_IDISO_2018_schemaV2.json"
}
MUON_HIGHPTID_RECO_FILE = {
    "2016" : "jsonpog-integration/POG/MUO/2016postVFP_UL/muon_Z.json",
    "2016UL_preVFP" : "jsonpog-integration/POG/MUO/2016preVFP_UL/ScaleFactors_Muon_highPt_RECO_2016_preVFP_schemaV2.json",
    "2016UL_postVFP" : "jsonpog-integration/POG/MUO/2016postVFP_UL/ScaleFactors_Muon_highPt_RECO_2016_schemaV2.json",
    "2017" : "jsonpog-integration/POG/MUO/2017_UL/ScaleFactors_Muon_highPt_RECO_2017_schemaV2.json",
    "2018" : "jsonpog-integration/POG/MUO/2018_UL/ScaleFactors_Muon_highPt_RECO_2018_schemaV2.json"
}

ELECTRON_ID_SF = {
    "2016" : "2016postVFP",
    "2016UL_preVFP" : "2016preVFP",
    "2016UL_postVFP" : "2016postVFP",
    "2017" : "2017",
    "2018" : "2018"
}
MUON_ID_SF = {
    "2016" : "2016preVFP_UL",
    "2016UL_preVFP" : "2016preVFP_UL",
    "2016UL_postVFP" : "2016postVFP_UL",
    "2017" : "2017_UL",
    "2018" : "2018_UL"
}

def electron_id_sf(events, year, central_only, input_collection, working_point = "none"):
    """
    See:
        - https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/EGM_electron_Run2_UL/EGM_electron_2017_UL.html
        - https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/electronExample.py
    """

    required_fields = [
        (input_collection, "eta"), (input_collection, "pt")
    ]

    missing_fields = awkward_utils.missing_fields(events, required_fields)

    evaluator = _core.CorrectionSet.from_file(misc_utils.expand_path(ELECTRON_ID_SF_FILE[year]))

    electrons = events[input_collection]

    # Flatten electrons then convert to numpy for compatibility with correctionlib
    n_electrons = awkward.num(electrons)
    electrons_flattened = awkward.flatten(electrons)

    ele_eta = numpy.clip(
        awkward.to_numpy(electrons_flattened.eta),
        -2.49999,
        2.49999 # SFs only valid up to eta 2.5
    )

    ele_pt = numpy.clip(
        awkward.to_numpy(electrons_flattened.pt),
        10.0, # SFs only valid for pT >= 10.0
        499.999 # and pT < 500.
    )

    # Calculate SF and syst
    variations = {}
    sf = evaluator["UL-Electron-ID-SF"].evalv(
            ELECTRON_ID_SF[year],
            "sf",
            working_point,
            ele_eta,
            ele_pt
    )
    variations["central"] = awkward.unflatten(sf, n_electrons)

    if not central_only:
        syst_vars = ["sfup", "sfdown"] 
        for syst_var in syst_vars:
            syst = evaluator["UL-Electron-ID-SF"].evalv(
                    ELECTRON_ID_SF[year],
                    syst_var,
                    working_point,
                    ele_eta,
                    ele_pt
            )
            if "up" in syst_var:
                syst_var_name = "up"
            elif "down" in syst_var:
                syst_var_name = "down"
            variations[syst_var_name] = awkward.unflatten(syst, n_electrons)

    for var in variations.keys():
        # Set SFs = 1 for leptons which are not applicable
        variations[var] = awkward.where(
                (electrons.pt < 10.0) | (electrons.pt >= 500.0) | (abs(electrons.eta) >= 2.5),
                awkward.ones_like(variations[var]),
                variations[var]
        )

    return variations




def muon_tightiso_sf(events, year, central_only, input_collection, working_point = "none"):
    """
    See:
        - https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/MUO_2017_UL_muon_Z.html
    """

    required_fields = [
         (input_collection, "eta"), (input_collection, "pt")
    ]
    missing_fields = awkward_utils.missing_fields(events, required_fields)
    logger.debug(year)
    evaluator = _core.CorrectionSet.from_file(misc_utils.expand_path(MUON_ID_SF_FILE[year]))
    # evaluator = _core.CorrectionSet.from_file(misc_utils.expand_path(MUON_ID_SF_FILE["2016UL_preVFP"]))

    muon = events[input_collection]

    # Flatten muon then convert to numpy for compatibility with correctionlib
    n_muon = awkward.num(muon)
    muon_flattened = awkward.flatten(muon)
    muon_pt = numpy.clip(
        awkward.to_numpy(muon_flattened.pt),
        15.0, # SFs only valid for pT >= 15.0
        2000 # and pT < Inf.
    )
    muon_eta = numpy.abs(numpy.clip(
        awkward.to_numpy(muon_flattened.eta),
        -2.39999,
        2.39999 # SFs only valid up to eta 2.4
    ))
    # Calculate SF and syst
    variations = {}


    sf = evaluator["NUM_TightRelIso_DEN_TightIDandIPCut"].evalv(
            MUON_ID_SF[year],
            muon_eta,
            muon_pt,
            "sf",
    )
    variations["central"] = awkward.unflatten(sf, n_muon)

    if not central_only:
        syst_vars = ["systup", "systdown"] 
        for syst_var in syst_vars:
            syst = evaluator["NUM_TightRelIso_DEN_TightIDandIPCut"].evalv(
                    MUON_ID_SF[year],
                    muon_eta,
                    muon_pt,
                    syst_var
            )
            if "up" in syst_var:
                syst_var_name = "up"
            elif "down" in syst_var:
                syst_var_name = "down"
            variations[syst_var_name] = awkward.unflatten(syst, n_muon)

    for var in variations.keys():
        # Set SFs = 1 for leptons which are not applicable
        variations[var] = awkward.where(
                (muon.pt < 15.0) |(abs(muon.eta) >= 2.4),
                awkward.ones_like(variations[var]),
                variations[var]
        )
    return variations

def muon_tightid_sf(events, year, central_only, input_collection, working_point = "none"):
    """
    See:
        - https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/MUO_2017_UL_muon_Z.html
    """

    required_fields = [
         (input_collection, "eta"), (input_collection, "pt")
    ]
    missing_fields = awkward_utils.missing_fields(events, required_fields)
    logger.debug(year)
    evaluator = _core.CorrectionSet.from_file(misc_utils.expand_path(MUON_ID_SF_FILE[year]))
    # evaluator = _core.CorrectionSet.from_file(misc_utils.expand_path(MUON_ID_SF_FILE["2016UL_preVFP"]))

    muon = events[input_collection]

    # Flatten muon then convert to numpy for compatibility with correctionlib
    n_muon = awkward.num(muon)
    muon_flattened = awkward.flatten(muon)
    muon_pt = numpy.clip(
        awkward.to_numpy(muon_flattened.pt),
        15.0, # SFs only valid for pT >= 15.0
        2000 # and pT < Inf.
    )
    muon_eta = numpy.abs(numpy.clip(
        awkward.to_numpy(muon_flattened.eta),
        -2.39999,
        2.39999 # SFs only valid up to eta 2.4
    ))
    # Calculate SF and syst
    variations = {}


    sf = evaluator["NUM_TightID_DEN_TrackerMuons"].evalv(
            MUON_ID_SF[year],
            muon_eta,
            muon_pt,
            "sf",
    )
    variations["central"] = awkward.unflatten(sf, n_muon)

    if not central_only:
        syst_vars = ["systup", "systdown"] 
        for syst_var in syst_vars:
            syst = evaluator["NUM_TightID_DEN_TrackerMuons"].evalv(
                    MUON_ID_SF[year],
                    muon_eta,
                    muon_pt,
                    syst_var
            )
            if "up" in syst_var:
                syst_var_name = "up"
            elif "down" in syst_var:
                syst_var_name = "down"
            variations[syst_var_name] = awkward.unflatten(syst, n_muon)

    for var in variations.keys():
        # Set SFs = 1 for leptons which are not applicable
        variations[var] = awkward.where(
                (muon.pt < 15.0) |(abs(muon.eta) >= 2.4),
                awkward.ones_like(variations[var]),
                variations[var]
        )
    return variations




def highptmuonid_sf(events, year, central_only, input_collection, working_point = "none"):
    required_fields = [
         (input_collection, "eta"), (input_collection, "pt"), (input_collection, "tunepRelPt")
    ]
    missing_fields = awkward_utils.missing_fields(events, required_fields)
    logger.debug(year)
    evaluator = _core.CorrectionSet.from_file(misc_utils.expand_path(MUON_HIGHPTID_SF_FILE[year]))
    # evaluator = _core.CorrectionSet.from_file(misc_utils.expand_path(MUON_ID_SF_FILE["2016UL_preVFP"]))

    muon = events[input_collection]

    # Flatten muon then convert to numpy for compatibility with correctionlib
    n_muon = awkward.num(muon)
    muon_flattened = awkward.flatten(muon)
    muon_pt = numpy.clip(
        awkward.to_numpy(muon_flattened.pt*muon_flattened.tunepRelPt),
        50.0, # SFs only valid for pT >= 15.0
        1999.999 # and pT < Inf.
    )
    muon_eta = numpy.abs(numpy.clip(
        awkward.to_numpy(muon_flattened.eta),
        -2.39999,
        2.39999 # SFs only valid up to eta 2.4
    ))
    # Calculate SF and syst
    variations = {}


    sf = evaluator["NUM_HighPtID_DEN_GlobalMuonProbes"].evalv(
            muon_eta,
            muon_pt,
            "nominal",
    )
    variations["central"] = awkward.unflatten(sf, n_muon)

    if not central_only:
        syst = evaluator["NUM_HighPtID_DEN_GlobalMuonProbes"].evalv(
                muon_eta,
                muon_pt,
                "syst"
        )

        variations_syst = awkward.unflatten(syst, n_muon)
        variations["up"] = variations["central"] + variations_syst
        variations["down"] = variations["central"] - variations_syst
    for var in variations.keys():
        variations[var] = awkward.where(
                (muon.pt*muon.tunepRelPt >= 2000.0) | (muon.pt*muon.tunepRelPt < 50.0) |(abs(muon.eta) >= 2.4),
                awkward.ones_like(variations[var]),
                variations[var]
        )
    return variations

def electron_reco_sf(events, year, central_only, input_collection, working_point = "none"):
    """
    See:
        - https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/EGM_electron_Run2_UL/EGM_electron_2017_UL.html
        - https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/electronExample.py
    """

    required_fields = [
        (input_collection, "eta"), (input_collection, "pt")
    ]

    missing_fields = awkward_utils.missing_fields(events, required_fields)

    evaluator = _core.CorrectionSet.from_file(misc_utils.expand_path(ELECTRON_ID_SF_FILE[year]))

    electrons = events[input_collection]

    # Flatten electrons then convert to numpy for compatibility with correctionlib
    n_electrons = awkward.num(electrons)
    electrons_flattened = awkward.flatten(electrons)

    ele_eta = numpy.clip(
        awkward.to_numpy(electrons_flattened.eta),
        -2.49999,
        2.49999 # SFs only valid up to eta 2.5
    )

    ele_lowpt = numpy.clip(
        awkward.to_numpy(electrons_flattened.pt),
        10.0, # SFs only valid for pT >= 10.0
        19.9999 # and pT < 500.
    )
    ele_highpt = numpy.clip(
        awkward.to_numpy(electrons_flattened.pt),
        20.0, # SFs only valid for pT >= 10.0
        499.999 # and pT < 500.
    )

    # Calculate SF and syst
    variations = {}
    sflow = evaluator["UL-Electron-ID-SF"].evalv(
            ELECTRON_ID_SF[year],
            "sf",
            "RecoBelow20",
            ele_eta,
            ele_lowpt
    )
    sfhigh = evaluator["UL-Electron-ID-SF"].evalv(
            ELECTRON_ID_SF[year],
            "sf",
            "RecoAbove20",
            ele_eta,
            ele_highpt
    )
    sf = sflow*sfhigh
    variations["central"] = awkward.unflatten(sf, n_electrons)

    if not central_only:
        syst_vars = ["sfup", "sfdown"] 
        for syst_var in syst_vars:
            systlow = evaluator["UL-Electron-ID-SF"].evalv(
                    ELECTRON_ID_SF[year],
                    syst_var,
                    "RecoBelow20",
                    ele_eta,
                    ele_lowpt
            )
            systhigh = evaluator["UL-Electron-ID-SF"].evalv(
            ELECTRON_ID_SF[year],
            syst_var,
            "RecoAbove20",
            ele_eta,
            ele_highpt
            )
            syst = systlow*systhigh
            if "up" in syst_var:
                syst_var_name = "up"
            elif "down" in syst_var:
                syst_var_name = "down"
            variations[syst_var_name] = awkward.unflatten(syst, n_electrons)

    for var in variations.keys():
        # Set SFs = 1 for leptons which are not applicable
        variations[var] = awkward.where(
                (electrons.pt < 10.0) | (electrons.pt >= 500.0) | (abs(electrons.eta) >= 2.5),
                awkward.ones_like(variations[var]),
                variations[var]
        )

    return variations



def highptmuonreco_sf(events, year, central_only, input_collection, working_point = "none"):
    required_fields = [
         (input_collection, "eta"), (input_collection, "pt"), (input_collection, "tunepRelPt"), (input_collection, "phi"), (input_collection, "mass")
    ]
    missing_fields = awkward_utils.missing_fields(events, required_fields)
    logger.debug(year)
    evaluator = _core.CorrectionSet.from_file(misc_utils.expand_path(MUON_HIGHPTID_SF_FILE[year]))
    # evaluator = _core.CorrectionSet.from_file(misc_utils.expand_path(MUON_ID_SF_FILE["2016UL_preVFP"]))

    muon = events[input_collection]
    vector.register_awkward()
    muon_4momentum=vector.obj(pt=muon.pt*muon.tunepRelPt, eta=muon.eta, phi=muon.phi, mass=muon.mass)
    # Flatten muon then convert to numpy for compatibility with correctionlib
    n_muon = awkward.num(muon)
    muon_flattened = awkward.flatten(muon)
    muon_pt = numpy.clip(
        awkward.to_numpy(awkward.flatten(muon_4momentum.p)),
        50.0, # SFs only valid for pT >= 15.0
        1999.999 # and pT < Inf.
    )
    muon_eta = numpy.abs(numpy.clip(
        awkward.to_numpy(muon_flattened.eta),
        -2.39999,
        2.39999 # SFs only valid up to eta 2.4
    ))
    # Calculate SF and syst
    variations = {}


    sf = evaluator["NUM_HighPtID_DEN_GlobalMuonProbes"].evalv(
            muon_eta,
            muon_pt,
            "nominal",
    )
    variations["central"] = awkward.unflatten(sf, n_muon)

    if not central_only:
        syst = evaluator["NUM_HighPtID_DEN_GlobalMuonProbes"].evalv(
                muon_eta,
                muon_pt,
                "syst"
        )

        variations_syst = awkward.unflatten(syst, n_muon)
        variations["up"] = variations["central"] + variations_syst
        variations["down"] = variations["central"] - variations_syst
    for var in variations.keys():
        variations[var] = awkward.where(
                (muon.pt*muon.tunepRelPt >= 1000.0) | (muon.pt*muon.tunepRelPt < 50.0) |(abs(muon.eta) >= 2.4),
                awkward.ones_like(variations[var]),
                variations[var]
        )
    return variations



    

