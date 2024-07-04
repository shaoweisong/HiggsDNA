import awkward

import logging
logger = logging.getLogger(__name__)

from higgs_dna.selections import object_selections
from higgs_dna.utils import misc_utils

DEFAULT_JETS = {
    "pt" : 25.0,
    "eta" : 2.4,
    "btagDeepFlavB" : 999
}

def select_jets(jets, options, clean, name = "none", tagger = None):
    """

    """
    options = misc_utils.update_dict(
        original = DEFAULT_JETS,
        new = options
    )

    tagger_name = "none" if tagger is None else tagger.name 
    bveto=jets.btagDeepFlavB<options["btagDeepFlavB"]
    standard_cuts = object_selections.select_objects(jets, options, clean, name, tagger)
    jetIDcut = (jets.jetId >= 6) 
    pujetIDcut = (jets.puId>=7)&(jets.pt<50)
    #  & (jets.btagDeepFlavB>=0.3040)

    all_cuts = (standard_cuts) & (jetIDcut) &(pujetIDcut)& (bveto)

    if tagger is not None:
        tagger.register_cuts(
            names = ["standard_cuts","jet ID cut","pujet ID cut","bveto","all cuts"],
            results = [standard_cuts,jetIDcut,pujetIDcut,bveto,all_cuts],
            cut_type = name
        )

    return all_cuts