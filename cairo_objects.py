from bson import SON
import math

SQUARE = [SON([('type','rectangle'),('args',(-.1,-.1,.2,.2))])]

SOLID_RED = SON([('type','SolidPattern'),('args',(1,0,0))])

SOLID_BLUE = SON([('type','SolidPattern'),('args',(0,1,0))])

CIRCLE = [SON([('type','arc'),('args',(0,0,.1,.0,2*math.pi))])]