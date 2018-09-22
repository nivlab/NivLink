import re
import numpy as np

def find_events(raw, message, event_id):
    """Find events from raw file.

    Parameters
    ----------
    raw : Raw object
        The raw data.
    message : string
        Pattern to search for in messages. Supports regex.
    event_id : int
        Condition identifier.
        
    Returns
    -------
    events : array, shape (n_events, 2) 
        All events that were found. The first column contains the event time 
        in samples and the second column contains the event id. 
    """
    assert isinstance(event_id, int)
    
    ## Identify matching messages.
    times = np.array([t for t, msg in raw.messages if re.search(message, msg) is not None])
    
    ## Make events.
    events = np.argwhere(np.in1d(raw.times, times))
    return np.column_stack((events, np.repeat(event_id,events.size)))

def concatenate_events(tup):
    """Concatenate events.
    
    Parameters
    ----------
    tup : sequence of events.
        Events to stack.

    Returns
    -------
    stacked : array, shape (n_events, 2)
        The events array formed by stacking the given events arrays.
    """
    
    ## Concatenate events.
    events = np.vstack(tup)
    
    ## Sort events.
    events = events[np.argsort(events[:,0])]
    
    ## Error-catching: detect duplicate events.
    if not np.all(np.diff(events[:,0])):
        raise ValueError('Duplicate events detected.')
    
    return events
    