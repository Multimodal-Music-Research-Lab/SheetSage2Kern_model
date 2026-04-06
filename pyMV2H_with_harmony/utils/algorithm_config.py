"""
The penalty assigned for insertion and deletion errors when performing alignment.
The default value of 1 leads to a reasonably fast, but not exhaustive
search through alignments.
"""
NON_ALIGNMENT_PENALTY = 1.

"""
The difference in onset time between two {@link mv2h.objects.Note}s for them to be
counted as a match.
Measured in milliseconds.
"""
ONSET_DELTA = 50

"""
The difference in time between beginning and end times of a {@link mv2h.objects.meter.Grouping}
for it to be counted as a match.

Measured in milliseconds.
"""
GROUPING_EPSILON = 50

"""
The difference in duration between two {@link mv2h.objects.Note}s for their value
to be counted as a match.

Measured in milliseconds.
"""
DURATION_DELTA = 100
