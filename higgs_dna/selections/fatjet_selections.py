import awkward

from higgs_dna.selections import object_selections
from higgs_dna.utils import misc_utils

DEFAULT_FATJETS = {
        "pt" : 150.,
        "eta" : 2.4,
        "ID": 6,
        "Hqqqq_vsQCDTop": -999 
}

def select_fatjets(fatjets, subjets, options, clean, name = "none", tagger = None): 
    """

    """
    options = misc_utils.update_dict(
        original = DEFAULT_FATJETS,
        new = options
    )
    standard_cuts = object_selections.select_objects(fatjets, options, clean, name, tagger)
    # can apply some additional cut
    fatjetIDcut = fatjets.jetId >= options["ID"]
    
    # add the H jet tagger for SL&FH channel((H3q+H4q+Hlvqq)/(H3q+H4q+Hlvqq+QCD+Top))
    H_jet_cut = fatjets.Hqqqq_vsQCDTop > options["Hqqqq_vsQCDTop"]
    H_jet_cut=awkward.fill_none(awkward.pad_none(H_jet_cut,1,axis=1),False,axis=-1)
    fatjetIDcut=awkward.fill_none(awkward.pad_none(fatjetIDcut,1,axis=1),False,axis=-1)
    # only for cut-based strategy, do subjet cut refer from bbWW analysis
    subjets_cut = ((awkward.fill_none(awkward.pad_none(subjets.pt,1,axis=1),-999,axis=-1)[awkward.fill_none(awkward.pad_none(fatjets.subJetIdx1,1,axis=1),0,axis=-1)]>0)==True)&((awkward.fill_none(awkward.pad_none(subjets.pt,1,axis=1),-999,axis=-1)[awkward.fill_none(awkward.pad_none(fatjets.subJetIdx2,1,axis=1),0,axis=-1)]>0)==True)
    standard_cuts=awkward.fill_none(awkward.pad_none(standard_cuts,1,axis=1),False,axis=-1)

    all_cuts = (standard_cuts) & (H_jet_cut) & (subjets_cut) & (fatjetIDcut)

    if tagger is not None:
        tagger.register_cuts(
            names = ["pteta cut","H_jet_cut","SubJet_cut", "fatjetIDcut"],
            results = [standard_cuts, H_jet_cut,subjets_cut, fatjetIDcut],
            cut_type = name
        )

    return all_cuts