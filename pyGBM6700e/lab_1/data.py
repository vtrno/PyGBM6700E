import pandas as pd

def select_view(dataframe:pd.DataFrame, view:str) -> pd.DataFrame:
    """
    Selects a view in a dataframe

    Parameters
    ----------
    dataframe : pd.DataFrame
        Source dataframe  
    view : str
        View name  

    Returns
    -------
    pd.DataFrame
        Dataframe only containing the selected view

    Raises
    ------
    ValueError
        View must be valid
    """
    available_views = pd.unique(dataframe['view'])
    if view not in available_views:
        raise ValueError('View must be in '+str(available_views.tolist()))

    return dataframe[dataframe['view'] == view]

def join_dataframes(dataframe_0:pd.DataFrame, dataframe_1:pd.DataFrame, key:str) -> pd.DataFrame:
    """
    Performs inner join operation on dataframes

    Parameters
    ----------
    dataframe_0 : pd.DataFrame
        First dataframe  
    dataframe_1 : pd.DataFrame
        Second dataframe  
    key : str
        Key on which to perform the join  

    Returns
    -------
    pd.DataFrame
        Final dataframe
    """
    df = pd.merge(dataframe_0, dataframe_1, 'inner', on=key)
    df = df.drop(columns=[x for x in df.columns if 'view' in x])
    return df