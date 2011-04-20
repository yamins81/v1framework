from bson import SON
import math

SOLID_RED = SON([('type','SolidPattern'),('args',(1,0,0))])

SOLID_BLUE = SON([('type','SolidPattern'),('args',(0,1,0))])

SQUARE = [SON([('type','rectangle'),('args',(-.1,-.1,.2,.2))])]

CIRCLE = [SON([('type','arc'),('args',(0,0,.1,.0,2*math.pi))])]


def solid_surface(r,g,b,a=1.0):
    return SON([('type','SolidPattern'),('args',(r,g,b,a))])

def linear_surface(x0,y0,x1,y1,stops):
    surface = SON([('type','LinearGradient'),('args',(x0,y0,x1,y1))])
    surface['color_stops'] = [SON([('type','rgba' if len(s) == 5 else 'rgb'),('args',s)]) for s in stops]
    return surface

def radial_surface(cx0, cy0, radius0, cx1, cy1, radius,stops):
    surface = SON([('type','RadialGradient'),('args',(cx0, cy0, radius0, cx1, cy1, radius))])
    surface['color_stops'] = [SON([('type','rgba' if len(s) == 5 else 'rgb'),('args',s)]) for s in stops]
    return surface

def square(side_length = None, radius = None, area = None, perimeter = None,lw=None):

    
    if area is not None:
        side_length = math.sqrt(area)
    if perimeter is not None:
        side_length = perimeter / 4
    if radius is not None:
        side_length = math.sqrt(2)*radius
        
    obj = [SON([('type','rectangle'),('args',(-side_length/2,-side_length/2,side_length,side_length))])]
    if lw:
        obj = [SON([('args',[lw]),('type','set_line_width')])] + obj 
    return obj

def disc(radius=None,area=None,perimeter = None, th=2*math.pi,lw=None):

    if area is not None:
    	radius = math.sqrt(area / math.pi)
    if perimeter is not None:
        radius = perimeter / 2*math.pi

    obj = [SON([('type','arc'),('args',(0,0,radius,.0,th))])]
    if lw:
        obj = [SON([('args',[lw]),('type','set_line_width')])] + obj 
    return obj

def polygon(n,radius = None, side_length = None, perimeter = None, area = None,lw = None):
    
    if perimeter is not None:
        side_length = perimeter / n
    
    if area is not None:
        side_length = 2 * math.sqrt(area * math.tan(math.pi/n) / n)
        
    if side_length is not None:
        radius = side_length / math.sqrt(2*(1-math.cos(2*math.pi/n)))
     
    points = [(radius*math.cos(2*math.pi/n*k),radius*math.sin(2*math.pi/n*k)) for k in range(n)]
    
    obj =  [SON([('type','move_to'),('args',points[0])])] + [SON([('type','line_to'),('args',p)]) for p in points[1:]] + [SON([('type','close_path'),('args',())])]
    if lw:
        obj = [SON([('args',[lw]),('type','set_line_width')])] + obj 
  
    return obj



