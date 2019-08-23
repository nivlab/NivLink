"""Wrapper for constants in edf_data.h"""

def invert_dict(d):
    new_dict = {}
    for k, v in d.items(): new_dict[v] = k
    return new_dict

event_codes = invert_dict(dict(
    
    NO_ITEMS        = 0,
    STARTPARSE      = 1,
    ENDPARSE        = 2,
    STARTBLINK      = 3,
    ENDBLINK        = 4,
    STARTSACC       = 5,
    ENDSACC         = 6,
    STARTFIX        = 7,
    ENDFIX          = 8,
    FIXUPDATE       = 9,
    BREAKPARSE      = 10,
    STARTSAMPLES    = 15,
    ENDSAMPLES      = 16,
    STARTEVENTS     = 17,
    ENDEVENTS       = 18,
    MESSAGEEVENT    = 24,
    BUTTONEVENT     = 25,
    INPUTEVENT      = 28,
    RECORDING       = 30,
    SAMPLES         = 200,
    LOST_DATA_EVENT = 0x3F
    
))