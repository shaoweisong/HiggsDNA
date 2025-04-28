import awkward as ak
eff = ak.sum(event.weight_central_no_lumi)/(br*xs)
# "xs":0.001,
# "bf":0.0009973889226000002,
#Since xs is in pb, we need to convert it to fb,0.001pb = 1fb, so the eff = ak.sum(event.weight_central_no_lumi)/0.0009973889226000002