def list_diff(listA, listB, mode="intersection"):
    """Calculate the difference between two lists based on the specified mode.
    
        Args:
            listA (list): The first list.
            listB (list): The second list.
            mode (str, optional): The mode to determine the difference. 
                Possible values are "intersection" (default), "union", or "diff".
    
        Returns:
            list: A list containing the elements based on the specified mode.
    """
    if mode == "intersection":
        ret = list(set(listA).intersection(set(listB)))
    elif mode == "union":
        ret = list(set(listA).union(set(listB)))
    elif mode == "diff":
        ret = list(set(listB).difference(set(listA)))
    
    return ret