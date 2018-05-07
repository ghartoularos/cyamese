import fcm

def cleanmarkers(markers):
    newmarkers = list()
    for i in markers:
        if i[-2:] == 'Dd': # Eliminate this 'Dd' tag if it has it
            i = i.rstrip('Dd')
        '''
        Sometimes marker name is left empty, and channel name is something like:
        "(Ba138)Dd", which is Barium138, so no marker. Do not change this. But others
        are like: "CD19(Ba138)Dd" which we *do* want to keep, so just check for open
        parentheses symbol.
        '''
        if '(' in i and i[0] != '(': 
            i = i.split('(')[0]
            
        # Capitalize everything for consistency
        i = i.upper()
        newmarkers.append(i)
    return newmarkers

def setmarkers(pathtofcs, files, ignorechan=[]):
    '''
    Define the set of markers that will be used, should be run through ALL files
    that will be used to train the model, not just those that are going through
    an individual batch. This list will become a global variable. A check should be
    made to check that the markerset is at least, say, 20 elements long.
    '''
    markerlist = list()
    for filename in files:
        # Load the channel names:
        markers = [i for i in fcm.loadFCS(pathtofcs+filename).channels]
        
        # Clean up the markers so they're consistent across files
        markers = cleanmarkers(markers)
        
        # Don't include anything that was set to be ignored
        markers = [i for i in markers if i not in ignorechan]
        
        markerlist.append(markers)
    markerset = set.intersection(*map(set,markerlist))
    assert len(markerset) > 20, \
    'Not enough common markers among the fcs files. Please check that channel names are' + \
    ' consistent between fcs files looking at the same marker or lower threshold. Note ' + \
    'lowering threshold could have negative effects on test set accuracy.'

    return markerset