import awkward
import numpy
import vector
import json
import logging
import io
import os
from higgs_dna.utils.metis_utils import do_cmd
import inspect
logger = logging.getLogger(__name__)
from higgs_dna.utils import misc_utils
from higgs_dna.constants import NOMINAL_TAG
from higgs_dna.utils.misc_utils import  get_HiggsDNA_base
from higgs_dna.utils.metis_utils import do_cmd

import sys

class Tagger():
    """
    Abstract base class for taggers
    :param name: Name to identify this tagger
    :type name: str
    :param options: options with selection info
    :type options: str, dict
    :param is_data: whether this tagger is being run on data
    :type is_data: bool
    :param year: which year this tagger is being run on
    :type year: str
    """
    def __init__(self, name = "tagger", options = {}, is_data = None, year = None, output_dir = None):
        self.name = name
        self.options = options
        self.is_data = is_data
        self.year = year
        self.output_dir = output_dir
        self.selection = {}
        self.events = {}
        self.cut_summary = {}

        self.options = misc_utils.load_config(options)


    def select(self, events):
        """
        Convenience function for running tagger in standalone contexts.

        :param events: awkward array of events
        :type events: awkward.Array
        :returns: awkward array of selected events
        :rtype: awkward.Array
        """
        self.current_syst = NOMINAL_TAG
        print("debugging taggerself.calculate_selection", events)
        selection, events_updated = self.calculate_selection(events)
        return events_updated[selection]


    def run(self, events, syst_tag = NOMINAL_TAG): 
        """
        Return dictionary of boolean arrays of events to be selected
        by this tagger for each systematic variation,
        along with updated dictionary of events containing any additional fields
        added by this tagger.
        :param events: sets of events to perform selection for (nominal + any independent collections for syst variations
        :type events: dict
        :return: boolean arrays of events to be selected by this tagger for each systematic variation, events dict with added fields computed by this tagger
        :rtype: dict, dict 
        """

        self.current_syst = syst_tag
   
        if not len(events) >= 1:
            logger.debug("[Tagger] %s : event set : %s : 0 events passed to tagger, skipping running this tagger." % (self.name, syst_tag))
            selection = awkward.ones_like(events, dtype=bool)
            self.selection[syst_tag] = selection

        else:
            print("tagger self.calculate_selection events", events)
            selection, events = self.calculate_selection(events)
            self.selection[syst_tag] = selection
            logger.debug("[Tagger] %s : event set : %s : %d (%d) events before (after) selection" % (self.name, syst_tag, len(selection), awkward.sum(selection)))

        return selection, events


    def calculate_selection(self, events): 
        """
        Abstract function that should be reimplemented for each tagger.
        """

        raise NotImplementedError()


    @staticmethod
    def get_range_cut(array, ranges):
        """
        Return mask corresponding to selecting elements
        in <array> that are within the ranges (inclusive)
        specified in <ranges>.
        :param array: array of quantity to be cut on
        :type array: awkward.Array
        :param ranges: list of 2-element lists corresponding to allowed ranges of variable in <array>
        :type ranges: list
        """
        cuts = []

        for range in ranges:
            if not len(range) == 2:
                message = "Range of allowed values should be 2 numbers ([lower bound, upper bound]), but you gave %s" % str(range)
                logger.exception(message)
                raise AssertionError(message)
            cut_low = array >= range[0]
            cut_high = array <= range[1]
            cuts.append(cut_low & cut_high)

        for idx, cut in enumerate(cuts):
            if idx == 0:
                final_cut = cut
            else:
                final_cut = final_cut | cut

        return final_cut


    def register_cuts(self, names, results, cut_type = "event"):
        """
        Record a given cut in the tagger instance.
        cut_type could be event-level (default) or object-level ("photon", "muon", etc)
        :param names: names to identify cuts
        :type names: list of str or str
        :param results: boolean arrays with results of applying given cut
        :type results: list of awkward.Array
        :param cut_type: flag to indicate the type of cut
        :type cut_type: str, optional
        """
        if cut_type not in self.cut_summary.keys():
            self.cut_summary[cut_type] = {}

        if not isinstance(names, list):
            names = [names]
        if not isinstance(results, list):
            results = [results]

        iter = 0
        for name, result in zip(names, results):
            iter += 1
            if awkward.count(result) > 0:
                individual_eff = float(awkward.sum(result)) / float(awkward.count(result))
            else:
                individual_eff = 0.
            self.cut_summary[cut_type][name] = {
                    "individual_eff" : float(individual_eff)
                    #TODO: add eff as N-1 cut
            }
            logger.debug("[Tagger] : %s, syst variation : %s, cut type : %s, cut : %s, indiviual efficiency : %.4f"% (self.name, self.current_syst, cut_type, name, individual_eff))
            # get the combined cuts and names
            if(iter==1):
                _tmp_cut = result
                _tmp_name = name
            else:
                _tmp_cut = numpy.logical_and(_tmp_cut, result)
                _tmp_name += "&" + name
            if awkward.count(_tmp_cut) > 0:
                ncandi_per_event = awkward.num(_tmp_cut[_tmp_cut==True],axis=-1) 
                candi_event=_tmp_cut[ncandi_per_event!=0]
                if type(candi_event) == bool:
                # for event selection level, like "at least one diphoton pair", _tmp_cut is 1D array, ncandi_per_event is the number of the events which contians at least one diphoton pair
                    combined_eff = float(ncandi_per_event) / float(len(_tmp_cut))
                    n_candi_event = ncandi_per_event
                else:
                    n_candi_event = len(candi_event)
                    combined_eff = float(n_candi_event) / float(len(_tmp_cut))
                    # combined_candieff = float(awkward.sum(_tmp_cut)) / float(awkward.count(_tmp_cut))
            else:
                
                combined_eff = 0.
                n_candi_event=0.
            self.cut_summary[cut_type][_tmp_name]={
                "combined eff": float(combined_eff)
            }
            logger.debug("cut(e-level) : %s\n combined_eff : %.4f\n event_num : %.4f"% (_tmp_name, combined_eff, n_candi_event))
            dic_eff = {'[object_eff] -'+cut_type+': '+name+' object efficiency':individual_eff,'[event_eff]   -'+cut_type+': '+_tmp_name+' event efficiency':combined_eff,'[event_number]   -'+cut_type+': '+_tmp_name+' event number' :n_candi_event}            
            output_dir = self.output_dir
            # Check if the file already exists
            if os.path.exists(output_dir+'/combined_eff.json'):
                file_size = os.path.getsize(output_dir+'/combined_eff.json')
                if file_size > 0:
                # Remove the closing bracket '}' from the existing file
                    with open(output_dir+'/combined_eff.json', 'rb+') as f:
                        f.seek(-1, os.SEEK_END)
                        f.truncate()
    
            # Open the file in append mode
            with open(output_dir+'/combined_eff.json', 'a') as f:
                # Get the file size
                file_size = os.path.getsize(output_dir+'/combined_eff.json')

                # Add opening bracket '[' if the file is empty
                if file_size == 0:
                    f.write('[')
                # Add comma ',' if the file is not empty
                else:
                    f.write(',')

                # Write the new JSON object
                json.dump(dic_eff, f, indent=4)

                # Add closing bracket ']' for the new object
                f.write(']')
            



    def get_summary(self):
        """
        Get dictionary of tag diagnostic info.
        :return: A dictionary of tag diagnostic info
        :rtype: dict
        """
        summary = {
                "options" : self.options,
                "cut_summary" : self.cut_summary
        }
        return summary
