import pandas as pd


def parse_windows_outlook_tracking(fp,
                                   organizer_attending=True,
                                   # which responses do we want?
                                    allowed_responses = ["Accepted"]):
    # this expects a tab delim file, like that returned by copy pasting
    # outlook calendar tracking into a text file
    # there is a "Name" column and a "Response" column...
    tracking = pd.read_csv(fp, sep="\t")
    return tracking[tracking["Response"].isin(allowed_responses)]["Name"]
    
    