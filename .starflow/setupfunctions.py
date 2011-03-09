import os
import cPickle as pickle
from traceback import print_exc
from starflow.utils import RecursiveFileList,CheckInOutFormulae, getpathalong, uniqify, ListUnion, IsPythonFile, IsDotPath, IsDir,listdir,IsFile,is_string_like,PathExists
from starflow.storage import FindMtime,StoredDocstring
from starflow.metadata import CombineSources,ProcessResources,ChooseImage,DEFAULT_GenerateAutomaticMetaData,metadatapath,opmetadatapath,ProcessMetaData


def get_live_modules(LiveModuleFilters):
    '''
    Function for filtering live modules that is fast by avoiding looking 
    through directories that will be irrelevant.
    '''
    FilteredModuleFiles = []
    Avoid = ['^RawData$','^Data$','^.svn$','.data$','^scrap$']
    FilterFn = lambda z,y : y.split('.')[-1] == 'py' and CheckInOutFormulae(z,y)
    for x in LiveModuleFilters.keys():
        Filter = lambda y : FilterFn(LiveModuleFilters[x],y) 
        FilteredModuleFiles += filter(Filter,RecursiveFileList(x,Avoid=Avoid))
    return FilteredModuleFiles



