"""
buddy_follow_me.py  —  Buddy Follow-Me robot simulation  (Ursina 3-D)
Visuals: detailed dog robot, articulated human figures, room-appropriate furniture

Engine : Ursina  (pip install ursina)

Camera controls
---------------
  W / S          — move forward / backward
  A / D          — strafe left / right
  Q / E          — move up / down
  Mouse drag     — look around  (hold RMB)
  Scroll wheel   — zoom (FOV)
  F              — toggle first-person / free-fly
  R              — reset camera to overview
  ESC            — quit
"""

import math, random, heapq
from ursina import *
from ursina.models.procedural.cylinder import Cylinder
from ursina.models.procedural.cone import Cone
from ursina.models.procedural.circle import Circle

# ─────────────────────────────────────────────
# SIMULATION PARAMETERS
# ─────────────────────────────────────────────
SIM_W, SIM_H = 900, 650
SCALE        = 0.012
WALL_H       = 0.55
FLOOR_Y      = 0.0
dt           = 1 / 60

robot_radius     = 15
max_speed        = 200
idle_speed       = 75

# PID — gentler gains, no derivative boost
Kp_distance = 2.2
Ki_distance = 0.02
Kd_distance = 0.8

target_radius    = 15
target_max_speed = 90

FOV_RADIUS    = 120
FOV_ANGLE     = math.radians(80)
FOV_ANGLE_SOFT= math.radians(115)
FOV_HALF      = FOV_ANGLE / 2
FOV_HALF_SOFT = FOV_ANGLE_SOFT / 2

LOST_GRACE_TIME     = 0.45
SEARCH_TIMEOUT      = 2.5
SEARCH_ROTATE_SPEED = 1.2   # slower scan rotation
SIGNAL_DURATION     = 1.5
ACK_DURATION        = 0.7
SCAN_BURST_DURATION = 2.2
PEEK_DIST           = 35
ATTENTION_DECAY_TIME= 4.0
ATTENTION_MIN       = 0.3
MAX_PATH_LEN        = 200
HUMAN_DIR_SMOOTH    = 0.06
HEEL_OFFSET_ANGLE   = math.radians(20)
HEEL_OFFSET_DIST    = 50   # further back so PID has breathing room

# Turn rate cap (radians/s) — limits how fast the dog can visually rotate
DOG_MAX_TURN_RATE = math.radians(200)   # 200 deg/s feels natural
# Visual rotation smoothing — separate from physics direction
_dog_visual_angle = 0.0

WALL_THICK = 10
DIV_X      = 530
DIV_Y      = 320
DOOR_W     = 80
DOOR_LIVING_DINING_Y  = 130
DOOR_LIVING_BEDROOM_Y = 490
DOOR_DINING_BEDROOM_X = 700

# ─────────────────────────────────────────────
# COORDINATE HELPERS
# ─────────────────────────────────────────────
def p2w(px_val, py_val):
    return Vec3(px_val * SCALE, FLOOR_Y, py_val * SCALE)

# ─────────────────────────────────────────────
# COLLISION GEOMETRY
# ─────────────────────────────────────────────
class Rect2:
    def __init__(self, x, y, w, h):
        self.left=x; self.top=y; self.right=x+w; self.bottom=y+h
        self.width=w; self.height=h
        self.centerx=x+w//2; self.centery=y+h//2
    def collidepoint(self,px2,py2):
        return self.left<=px2<=self.right and self.top<=py2<=self.bottom
    def inflate(self,dx,dy):
        return Rect2(self.left-dx,self.top-dy,self.width+2*dx,self.height+2*dy)

def _rect(x,y,w,h): return Rect2(x,y,w,h)

wall_rects = [
    _rect(0,0,SIM_W,WALL_THICK),
    _rect(0,SIM_H-WALL_THICK,SIM_W,WALL_THICK),
    _rect(0,0,WALL_THICK,SIM_H),
    _rect(SIM_W-WALL_THICK,0,WALL_THICK,SIM_H),
    _rect(DIV_X,WALL_THICK,WALL_THICK,DOOR_LIVING_DINING_Y-DOOR_W//2-WALL_THICK),
    _rect(DIV_X,DOOR_LIVING_DINING_Y+DOOR_W//2,WALL_THICK,
          DOOR_LIVING_BEDROOM_Y-DOOR_W//2-(DOOR_LIVING_DINING_Y+DOOR_W//2)),
    _rect(DIV_X,DOOR_LIVING_BEDROOM_Y+DOOR_W//2,WALL_THICK,
          SIM_H-WALL_THICK-(DOOR_LIVING_BEDROOM_Y+DOOR_W//2)),
    _rect(DIV_X+WALL_THICK,DIV_Y,DOOR_DINING_BEDROOM_X-DOOR_W//2-DIV_X-WALL_THICK,WALL_THICK),
    _rect(DOOR_DINING_BEDROOM_X+DOOR_W//2,DIV_Y,
          SIM_W-WALL_THICK-(DOOR_DINING_BEDROOM_X+DOOR_W//2),WALL_THICK),
]

obstacles_rects = [
    _rect(180,15,170,35),   # 0 TV unit
    _rect(140,230,230,55),  # 1 sofa
    _rect(175,300,110,45),  # 2 coffee table
    _rect(15,280,45,160),   # 3 bookshelf
    _rect(15,540,90,80),    # 4 armchair
    _rect(615,185,200,90),  # 5 dining table
    _rect(820,20,55,100),   # 6 kitchen counter
    _rect(700,490,170,120), # 7 bed
    _rect(830,335,50,140),  # 8 wardrobe
    _rect(540,570,70,60),   # 9 desk
]

all_collidables = wall_rects + obstacles_rects

def circle_rect_col(cx,cy,radius,rect):
    clx=max(rect.left,min(cx,rect.right))
    cly=max(rect.top, min(cy,rect.bottom))
    return (cx-clx)**2+(cy-cly)**2<radius*radius

# ─────────────────────────────────────────────
# A*  PATHFINDING
# ─────────────────────────────────────────────
NAV_CELL=7; NAV_CLEARANCE=13; WALL_CLEARANCE=8
nav_cols=SIM_W//NAV_CELL; nav_rows=SIM_H//NAV_CELL

def _build_nav_grid():
    g=[[True]*nav_cols for _ in range(nav_rows)]
    for r in range(nav_rows):
        for c in range(nav_cols):
            wx=c*NAV_CELL+NAV_CELL//2; wy=r*NAV_CELL+NAV_CELL//2
            for ob in wall_rects:
                if ob.inflate(WALL_CLEARANCE,WALL_CLEARANCE).collidepoint(wx,wy):
                    g[r][c]=False; break
            if not g[r][c]: continue
            for ob in obstacles_rects:
                if ob.inflate(NAV_CLEARANCE,NAV_CLEARANCE).collidepoint(wx,wy):
                    g[r][c]=False; break
    return g

nav_grid=_build_nav_grid()

def _build_cost_map():
    PENALTY_RADIUS=4
    cost=[[1.0]*nav_cols for _ in range(nav_rows)]
    for r in range(nav_rows):
        for c in range(nav_cols):
            if not nav_grid[r][c]: continue
            md=float('inf')
            for dr in range(-PENALTY_RADIUS,PENALTY_RADIUS+1):
                for dc in range(-PENALTY_RADIUS,PENALTY_RADIUS+1):
                    nr2,nc2=r+dr,c+dc
                    if 0<=nr2<nav_rows and 0<=nc2<nav_cols and not nav_grid[nr2][nc2]:
                        d=math.hypot(dr,dc)
                        if d<md: md=d
            if md<PENALTY_RADIUS: cost[r][c]=1.0+2.0*(1.0-md/PENALTY_RADIUS)
    return cost

nav_cost=_build_cost_map()

def world_to_cell(wx,wy):
    c=int(wx//NAV_CELL); r=int(wy//NAV_CELL)
    return max(0,min(nav_cols-1,c)),max(0,min(nav_rows-1,r))

def cell_to_world(c,r):
    return c*NAV_CELL+NAV_CELL//2,r*NAV_CELL+NAV_CELL//2

def astar(start_w,goal_w):
    sc,sr=world_to_cell(*start_w); gc,gr=world_to_cell(*goal_w)
    for _ in range(2):
        tr,tc=(sr,sc) if _==0 else (gr,gc)
        if not nav_grid[tr][tc]:
            for radius in range(1,8):
                found=False
                for dr in range(-radius,radius+1):
                    for dc in range(-radius,radius+1):
                        nr,nc=tr+dr,tc+dc
                        if 0<=nr<nav_rows and 0<=nc<nav_cols and nav_grid[nr][nc]:
                            if _==0: sr,sc=nr,nc
                            else:    gr,gc=nr,nc
                            found=True; break
                    if found: break
                if found: break
    if (sc,sr)==(gc,gr): return [goal_w]
    open_heap=[]; heapq.heappush(open_heap,(0.0,sc,sr))
    came_from={}; g_score={(sc,sr):0.0}
    nbrs=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while open_heap:
        _,cc,cr=heapq.heappop(open_heap)
        if (cc,cr)==(gc,gr):
            path_cells=[]; node=(gc,gr)
            while node in came_from:
                path_cells.append(node); node=came_from[node]
            path_cells.reverse()
            wp=[cell_to_world(c,r) for c,r in path_cells]; wp.append(goal_w)
            return wp
        for dc,dr in nbrs:
            nc2,nr2=cc+dc,cr+dr
            if not(0<=nr2<nav_rows and 0<=nc2<nav_cols): continue
            if not nav_grid[nr2][nc2]: continue
            mc=(math.sqrt(2) if dc!=0 and dr!=0 else 1.0)*nav_cost[nr2][nc2]
            ng=g_score[(cc,cr)]+mc
            if ng<g_score.get((nc2,nr2),float('inf')):
                g_score[(nc2,nr2)]=ng; came_from[(nc2,nr2)]=(cc,cr)
                h=math.hypot(nc2-gc,nr2-gr)
                heapq.heappush(open_heap,(ng+h,nc2,nr2))
    return []

def _los_clear(a,b):
    steps=max(int(math.hypot(b[0]-a[0],b[1]-a[1])/(NAV_CELL*0.7)),2)
    for t in range(1,steps+1):
        px2=a[0]+(b[0]-a[0])*t/steps; py2=a[1]+(b[1]-a[1])*t/steps
        for ob in wall_rects:
            if ob.inflate(WALL_CLEARANCE,WALL_CLEARANCE).collidepoint(px2,py2): return False
        for ob in obstacles_rects:
            if ob.inflate(NAV_CLEARANCE//2,NAV_CLEARANCE//2).collidepoint(px2,py2): return False
    return True

def find_peek_position(rx,ry,sx,sy):
    best_corner=None; best_score=float('inf')
    for ob in all_collidables:
        clr=WALL_CLEARANCE if ob in wall_rects else NAV_CLEARANCE//2
        exp=ob.inflate(clr,clr); seg_blocked=False
        steps=max(int(math.hypot(sx-rx,sy-ry)/10),2)
        for t in range(1,steps):
            px2=rx+(sx-rx)*t/steps; py2=ry+(sy-ry)*t/steps
            if exp.collidepoint(px2,py2): seg_blocked=True; break
        if not seg_blocked: continue
        corners=[(ob.left,ob.top),(ob.right,ob.top),(ob.left,ob.bottom),(ob.right,ob.bottom)]
        for cx2,cy2 in corners:
            odx=cx2-ob.centerx; ody=cy2-ob.centery
            om=math.hypot(odx,ody) or 1
            px3=cx2+(odx/om)*PEEK_DIST; py3=cy2+(ody/om)*PEEK_DIST
            if not(WALL_THICK+5<px3<SIM_W-WALL_THICK-5 and WALL_THICK+5<py3<SIM_H-WALL_THICK-5): continue
            if any(circle_rect_col(px3,py3,robot_radius,o) for o in all_collidables): continue
            if not _los_clear((px3,py3),(sx,sy)): continue
            d=math.hypot(px3-rx,py3-ry)
            if d<best_score: best_score=d; best_corner=(px3,py3)
    return best_corner

def ray_hits_obstacle(rx,ry,tx,ty,obs):
    for ob in obs:
        for t in [i/12 for i in range(1,12)]:
            if ob.collidepoint(rx+(tx-rx)*t,ry+(ty-ry)*t): return True
    return False

def safe_spawn(bounds,radius,max_tries=200):
    for _ in range(max_tries):
        sx=random.uniform(bounds.left+radius+2,bounds.right-radius-2)
        sy=random.uniform(bounds.top+radius+2, bounds.bottom-radius-2)
        if not any(circle_rect_col(sx,sy,radius+4,o) for o in all_collidables):
            return sx,sy
    return bounds.centerx,bounds.centery

def safe_audio_guess(gx, gy, clearance=18, max_tries=24):
    """Nudge an audio-guess point out of any obstacle/wall, trying nearby offsets."""
    CHECK_R = clearance
    if not any(circle_rect_col(gx, gy, CHECK_R, o) for o in all_collidables):
        return gx, gy  # already clear
    # Spiral outward to find a clear spot
    for step in range(1, max_tries + 1):
        radius_try = step * 8
        for angle_off in range(0, 360, 30):
            a = math.radians(angle_off)
            nx = gx + math.cos(a) * radius_try
            ny = gy + math.sin(a) * radius_try
            nx = max(WALL_THICK + 5, min(SIM_W - WALL_THICK - 5, nx))
            ny = max(WALL_THICK + 5, min(SIM_H - WALL_THICK - 5, ny))
            if not any(circle_rect_col(nx, ny, CHECK_R, o) for o in all_collidables):
                return nx, ny
    # Fallback: clamp to sim bounds only
    return (max(WALL_THICK + 5, min(SIM_W - WALL_THICK - 5, gx)),
            max(WALL_THICK + 5, min(SIM_H - WALL_THICK - 5, gy)))

# ─────────────────────────────────────────────
# ROOM / TASK DATA
# ─────────────────────────────────────────────
rooms={
    "living":  {"bounds":_rect(WALL_THICK,WALL_THICK,DIV_X-WALL_THICK,SIM_H-2*WALL_THICK),
                "tasks":[("watching TV",(265,80),(2,5)),("reading",(250,360),(2,5)),
                          ("relaxing",(250,255),(1,3)),("stretching",(390,430),(1,3)),
                          ("looking outside",(460,550),(1,3))]},
    "dining":  {"bounds":_rect(DIV_X+WALL_THICK,WALL_THICK,SIM_W-DIV_X-2*WALL_THICK,DIV_Y-WALL_THICK),
                "tasks":[("eating",(715,220),(3,6)),("making coffee",(840,70),(1,3)),
                          ("reading paper",(715,250),(2,5)),("checking phone",(620,100),(1,3))]},
    "bedroom": {"bounds":_rect(DIV_X+WALL_THICK,DIV_Y+WALL_THICK,SIM_W-DIV_X-2*WALL_THICK,SIM_H-DIV_Y-2*WALL_THICK),
                "tasks":[("sleeping",(785,550),(4,8)),("getting dressed",(848,410),(1,3)),
                          ("resting",(785,580),(2,4)),("at desk",(575,590),(1,3))]},
}
room_order=["living","dining","bedroom"]
door_paths={
    ("living","dining"):  [(DIV_X-30,DOOR_LIVING_DINING_Y),(DIV_X+30,DOOR_LIVING_DINING_Y)],
    ("dining","living"):  [(DIV_X+30,DOOR_LIVING_DINING_Y),(DIV_X-30,DOOR_LIVING_DINING_Y)],
    ("living","bedroom"): [(DIV_X-30,DOOR_LIVING_BEDROOM_Y),(DIV_X+30,DOOR_LIVING_BEDROOM_Y)],
    ("bedroom","living"): [(DIV_X+30,DOOR_LIVING_BEDROOM_Y),(DIV_X-30,DOOR_LIVING_BEDROOM_Y)],
    ("dining","bedroom"): [(DOOR_DINING_BEDROOM_X,DIV_Y-30),(DOOR_DINING_BEDROOM_X,DIV_Y+30)],
    ("bedroom","dining"): [(DOOR_DINING_BEDROOM_X,DIV_Y+30),(DOOR_DINING_BEDROOM_X,DIV_Y-30)],
}
def pick_next_room(current): return random.choice([r for r in room_order if r!=current])
def pick_task(room_name):
    name,pos,dwell=random.choice(rooms[room_name]["tasks"])
    return name,pos,random.uniform(*dwell)

# ═════════════════════════════════════════════
# URSINA APP
# ═════════════════════════════════════════════
app=Ursina(title="Buddy Follow-Me — Realistic 3D",borderless=False)
window.fps_counter.enabled=True
window.exit_button.visible=False

# ── Lighting ─────────────────────────────────
AmbientLight(color=color.rgba32(135,130,125,255))          # muted neutral ambient
dl=DirectionalLight(color=color.rgba32(220,205,178,255))   # warm afternoon sun (softer)
dl.look_at(Vec3(0.6,-1.0,0.8))
dl2=DirectionalLight(color=color.rgba32(150,168,210,70))   # cool sky fill (dimmer)
dl2.look_at(Vec3(-0.5,-0.6,-0.3))
# Room lights
PointLight(position=p2w(265,325)+Vec3(0,3,0),  color=color.rgba32(230,210,170,160))  # living
PointLight(position=p2w(715,160)+Vec3(0,3,0),  color=color.rgba32(225,220,190,140))  # dining
PointLight(position=p2w(715,490)+Vec3(0,2.5,0),color=color.rgba32(175,185,225,120))  # bedroom

# ── FLOOR — hardwood planks effect ───────────
floor=Entity(
    model='plane',
    scale=(SIM_W*SCALE,1,SIM_H*SCALE),
    position=Vec3(SIM_W*SCALE/2,-0.01,SIM_H*SCALE/2),
    color=color.rgba32(148,112,72),   # muted warm timber
)
# Subtle room carpet/rug overlays
def floor_rug(rect2,col,h=0.002):
    cx=(rect2.left+rect2.right)/2; cy=(rect2.top+rect2.bottom)/2
    Entity(model='cube',
           position=Vec3(cx*SCALE,h,cy*SCALE),
           scale=(rect2.width*SCALE*0.82,0.004,rect2.height*SCALE*0.82),
           color=col)

floor_rug(rooms["living"]["bounds"],  color.rgba32(55,72,118,170))   # muted navy rug
floor_rug(rooms["dining"]["bounds"],  color.rgba32(140,105,68,60))   # bare floor tint
floor_rug(rooms["bedroom"]["bounds"], color.rgba32(148,100,128,130)) # dusty mauve rug

# ── WALLS — plaster white with baseboard ─────
WALL_COLOR    = color.rgba32(235,230,220)   # off-white plaster
BASEBOARD_COL = color.rgba32(250,248,240)   # bright white trim

def make_wall(rect2,col=WALL_COLOR):
    cx=(rect2.left+rect2.right)/2; cy=(rect2.top+rect2.bottom)/2
    # Main wall body
    Entity(model='cube',
           position=Vec3(cx*SCALE,WALL_H/2,cy*SCALE),
           scale=(rect2.width*SCALE,WALL_H,rect2.height*SCALE),
           color=col)
    # Baseboard strip at bottom
    Entity(model='cube',
           position=Vec3(cx*SCALE,0.025,cy*SCALE),
           scale=(rect2.width*SCALE+0.01,0.05,rect2.height*SCALE+0.01),
           color=BASEBOARD_COL)

for wr in wall_rects:
    make_wall(wr)

# ── CEILING (semi-transparent so camera can see in) ──
Entity(model='plane',
       scale=(SIM_W*SCALE,1,SIM_H*SCALE),
       position=Vec3(SIM_W*SCALE/2,WALL_H+0.01,SIM_H*SCALE/2),
       rotation_x=180,
       color=color.rgba32(245,243,238,60))   # nearly invisible ceiling

# Door gap markers — polished brass threshold strips
def door_strip(x,y,w,h):
    Entity(model='cube',
           position=Vec3((x+w/2)*SCALE,0.005,(y+h/2)*SCALE),
           scale=(w*SCALE,0.012,h*SCALE),
           color=color.rgba32(200,170,80))

door_strip(DIV_X-4,DOOR_LIVING_DINING_Y-DOOR_W//2,  18,DOOR_W)
door_strip(DIV_X-4,DOOR_LIVING_BEDROOM_Y-DOOR_W//2, 18,DOOR_W)
door_strip(DOOR_DINING_BEDROOM_X-DOOR_W//2,DIV_Y-4, DOOR_W,18)

# ═════════════════════════════════════════════
# FURNITURE BUILDER
# Replaces plain cubes with multi-part objects
# ═════════════════════════════════════════════

def _box(parent,pos,scale,col):
    """Convenience: child cube entity."""
    return Entity(parent=parent,model='cube',position=pos,scale=scale,color=col)

def _cyl(parent,pos,scale,col,res=8):
    return Entity(parent=parent,model=Cylinder(resolution=res,height=1),
                  position=pos,scale=scale,color=col)

def _sph(parent,pos,sc,col):
    return Entity(parent=parent,model='sphere',position=pos,scale=sc,color=col)

# ── Obstacle 0: TV Console + TV ──────────────
def make_tv_unit(rect):
    cx=(rect.left+rect.right)/2; cy=(rect.top+rect.bottom)/2
    root=Entity(position=Vec3(cx*SCALE,0,cy*SCALE))
    w=rect.width*SCALE; d=rect.height*SCALE; h=WALL_H*0.45
    # Console body — dark walnut
    _box(root,Vec3(0,h/2,0),(w,h,d),color.rgba32(60,38,20))
    # Console legs (4 small)
    for lx,lz in [(w*0.4,d*0.35),(w*0.4,-d*0.35),(-w*0.4,d*0.35),(-w*0.4,-d*0.35)]:
        _box(root,Vec3(lx,0.01,lz),(0.025,0.04,0.025),color.rgba32(40,25,10))
    # Drawer faces
    for dx2 in [-w*0.25,0,w*0.25]:
        _box(root,Vec3(dx2,h*0.55,d*0.52),(w*0.28,h*0.22,0.01),color.rgba32(80,52,28))
        _box(root,Vec3(dx2,h*0.55,d*0.53),(0.025,0.01,0.005),color.rgba32(180,150,60))  # handle
    # TV screen — large flat black rectangle above console
    tv_h=h*2.5; tv_w=w*0.88
    _box(root,Vec3(0,h+tv_h/2+0.02,0),(tv_w,tv_h,0.04),color.rgba32(12,12,14))  # bezel
    _box(root,Vec3(0,h+tv_h/2+0.02,0.022),(tv_w*0.93,tv_h*0.90,0.01),color.rgba32(30,40,80))  # screen
    # TV stand
    _box(root,Vec3(0,h+0.015,0),(tv_w*0.12,0.03,d*0.4),color.rgba32(40,40,40))

# ── Obstacle 1: Sofa ─────────────────────────
def make_sofa(rect):
    cx=(rect.left+rect.right)/2; cy=(rect.top+rect.bottom)/2
    root=Entity(position=Vec3(cx*SCALE,0,cy*SCALE))
    w=rect.width*SCALE; d=rect.height*SCALE
    FABRIC=color.rgba32(80,60,110)      # deep slate-purple
    WOOD  =color.rgba32(90,55,25)
    # Seat cushions — centred
    for sx2 in [-w*0.3,0,w*0.3]:
        _box(root,Vec3(sx2,0.18,0),(w*0.28,0.12,d*0.65),FABRIC)
    # Back cushions — face toward TV (small Z = top wall) so back is at +d side
    for sx2 in [-w*0.3,0,w*0.3]:
        _box(root,Vec3(sx2,0.34,d*0.28),(w*0.28,0.22,0.10),color.rgba32(90,68,125))
    # Base frame
    _box(root,Vec3(0,0.08,0),(w,0.16,d),WOOD)
    # Arm rests
    _box(root,Vec3( w*0.48,0.28,0),(0.10,0.30,d),color.rgba32(70,48,100))
    _box(root,Vec3(-w*0.48,0.28,0),(0.10,0.30,d),color.rgba32(70,48,100))
    # Back rest panel — behind sofa (toward larger Z, away from TV)
    _box(root,Vec3(0,0.40,d*0.35),(w,0.08,0.10),WOOD)
    # Four legs
    for lx,lz in [(w*0.42,d*0.38),(w*0.42,-d*0.38),(-w*0.42,d*0.38),(-w*0.42,-d*0.38)]:
        _box(root,Vec3(lx,0.03,lz),(0.04,0.06,0.04),WOOD)

# ── Obstacle 2: Coffee Table ─────────────────
def make_coffee_table(rect):
    cx=(rect.left+rect.right)/2; cy=(rect.top+rect.bottom)/2
    root=Entity(position=Vec3(cx*SCALE,0,cy*SCALE))
    w=rect.width*SCALE; d=rect.height*SCALE
    OAK=color.rgba32(160,110,55); METAL=color.rgba32(80,80,85)
    # Table top
    _box(root,Vec3(0,0.24,0),(w,0.05,d),OAK)
    # Lower shelf
    _box(root,Vec3(0,0.10,0),(w*0.8,0.03,d*0.8),color.rgba32(140,95,45))
    # Metal legs (hairpin style — 2 sets)
    for lx in [w*0.38,-w*0.38]:
        for lz in [d*0.35,-d*0.35]:
            _box(root,Vec3(lx,0.12,lz),(0.015,0.24,0.015),METAL)
    # Book / remote on top
    _box(root,Vec3(-w*0.2,0.272,0),(w*0.25,0.02,d*0.35),color.rgba32(180,40,40))
    _box(root,Vec3( w*0.22,0.272,d*0.1),(0.06,0.015,0.12),color.rgba32(30,30,30))

# ── Obstacle 3: Bookshelf ────────────────────
def make_bookshelf(rect):
    cx=(rect.left+rect.right)/2; cy=(rect.top+rect.bottom)/2
    # Bookshelf is against the left wall — rotate 90 deg so it opens toward +X (into room)
    root=Entity(position=Vec3(cx*SCALE,0,cy*SCALE), rotation_y=90)
    # After rotation: local X = world Z, local Z = world -X
    # w (rect.width=45) becomes the shallow depth, d (rect.height=160) becomes the width along wall
    w=rect.width*SCALE; d=rect.height*SCALE; H=WALL_H*0.95
    WOOD=color.rgba32(100,65,28)
    # Frame
    _box(root,Vec3(0,H/2,0),(d,H,w),color.rgba32(85,55,22))
    # Back panel — against the wall (local +Z = world -X = toward left wall)
    _box(root,Vec3(0,H/2,w*0.45),(d*0.94,H*0.96,0.01),color.rgba32(120,80,35))
    # Shelves (4)
    for sh in [0.15,0.32,0.49,0.66]:
        _box(root,Vec3(0,sh,0),(d*0.92,0.025,w*0.90),WOOD)
    # Books — random coloured spines along the shelf length
    book_cols=[(180,40,40),(40,80,180),(40,140,60),(160,120,40),(120,40,150),(40,140,150)]
    bz=-d*0.42
    for i in range(12):
        bc=color.rgba32(*book_cols[i%len(book_cols)])
        bw=random.uniform(0.016,0.028)
        shelf_y=0.15+((i//4)%4)*0.17+0.06
        _box(root,Vec3(bz+bw/2,shelf_y,0),(bw,0.11,w*0.75),bc)
        bz+=bw+0.005
        if bz>d*0.42: bz=-d*0.42

# ── Obstacle 4: Armchair ─────────────────────
def make_armchair(rect):
    cx=(rect.left+rect.right)/2; cy=(rect.top+rect.bottom)/2
    root=Entity(position=Vec3(cx*SCALE,0,cy*SCALE))
    w=rect.width*SCALE; d=rect.height*SCALE
    FAB =color.rgba32(150,85,40)   # rust/tan leather
    DARK=color.rgba32(60,38,18)
    _box(root,Vec3(0,0.08,0),(w,0.16,d),DARK)           # base
    _box(root,Vec3(0,0.20, d*0.05),(w*0.80,0.16,d*0.68),FAB)  # seat
    _box(root,Vec3(0,0.38,-d*0.29),(w*0.80,0.32,0.12),  color.rgba32(165,95,48))  # back
    _box(root,Vec3( w*0.43,0.30,0),(0.10,0.24,d),FAB)   # arm L
    _box(root,Vec3(-w*0.43,0.30,0),(0.10,0.24,d),FAB)   # arm R
    for lx,lz in [(w*0.38,d*0.38),(w*0.38,-d*0.38),(-w*0.38,d*0.38),(-w*0.38,-d*0.38)]:
        _box(root,Vec3(lx,0.03,lz),(0.04,0.06,0.04),DARK)

# ── Obstacle 5: Dining Table + 4 Chairs ──────
def make_dining_table(rect):
    cx=(rect.left+rect.right)/2; cy=(rect.top+rect.bottom)/2
    root=Entity(position=Vec3(cx*SCALE,0,cy*SCALE))
    w=rect.width*SCALE; d=rect.height*SCALE
    OAK=color.rgba32(175,120,55); METAL=color.rgba32(70,70,75)
    # Table top
    _box(root,Vec3(0,0.40,0),(w*0.92,0.05,d*0.88),OAK)
    # Pedestal legs (X shape)
    for ang in [0,90]:
        a=math.radians(ang)
        e=Entity(parent=root,model='cube',
                 position=Vec3(0,0.20,0),
                 rotation_y=ang,
                 scale=(w*0.08,0.40,d*0.70),
                 color=METAL)
    # Foot cross-bars
    _box(root,Vec3(0,0.04,0),(w*0.60,0.04,d*0.12),METAL)
    _box(root,Vec3(0,0.04,0),(w*0.12,0.04,d*0.60),METAL)
    # Place settings (plates)
    for px2,pz2 in [(w*0.28,0),(-w*0.28,0),(0,d*0.30),(0,-d*0.30)]:
        _cyl(root,Vec3(px2,0.43,pz2),(0.09,0.015,0.09),color.rgba32(245,242,235),16)
    # Chairs (4)
    CHAIR_F=color.rgba32(90,60,20); SEAT=color.rgba32(155,105,55)
    for chx,chz,chy_rot in [(w*0.55,0,90),(-w*0.55,0,-90),(0,d*0.58,0),(0,-d*0.58,180)]:
        ch=Entity(parent=root,position=Vec3(chx,0,chz),rotation_y=chy_rot)
        _box(ch,Vec3(0,0.23,0),(0.18,0.06,0.18),SEAT)          # seat
        _box(ch,Vec3(0,0.40,-0.07),(0.18,0.28,0.05),CHAIR_F)   # back
        for cx2,cz2 in [(0.07,0.07),(0.07,-0.07),(-0.07,0.07),(-0.07,-0.07)]:
            _box(ch,Vec3(cx2,0.10,cz2),(0.025,0.20,0.025),CHAIR_F)

# ── Obstacle 6: Kitchen Counter ──────────────
def make_kitchen_counter(rect):
    cx=(rect.left+rect.right)/2; cy=(rect.top+rect.bottom)/2
    root=Entity(position=Vec3(cx*SCALE,0,cy*SCALE))
    w=rect.width*SCALE; d=rect.height*SCALE
    BASE=color.rgba32(220,215,210); TOP=color.rgba32(90,90,95)   # white cabinets, granite top
    _box(root,Vec3(0,0.20,0),(w,0.40,d),BASE)                   # cabinet body
    _box(root,Vec3(0,0.42,0),(w*1.02,0.04,d*1.02),TOP)          # countertop overhang
    # Cabinet doors with handles
    for dx2 in [-w*0.25,w*0.25]:
        _box(root,Vec3(dx2,0.20,d*0.52),(w*0.44,0.34,0.015),color.rgba32(230,228,224))
        _box(root,Vec3(dx2,0.20,d*0.535),(0.04,0.02,0.008),color.rgba32(160,155,150))
    # Sink
    _box(root,Vec3(-w*0.1,0.435,0),(w*0.35,0.03,d*0.65),color.rgba32(190,195,200))
    _cyl(root,Vec3(-w*0.1,0.46,0),(0.04,0.06,0.04),color.rgba32(160,160,165),8)  # tap
    # Coffee machine
    _box(root,Vec3(w*0.3,0.52,0),(0.08,0.18,0.09),color.rgba32(20,20,20))
    _box(root,Vec3(w*0.3,0.54,-d*0.15),(0.09,0.08,0.06),color.rgba32(140,35,35))  # coffee pod
    # Upper wall cabinet
    _box(root,Vec3(0,0.78,0),(w,0.28,d*0.45),BASE)
    _box(root,Vec3(0,0.78,d*0.235),(w*0.98,0.26,0.015),color.rgba32(230,228,224))

# ── Obstacle 7: Bed ──────────────────────────
def make_bed(rect):
    cx=(rect.left+rect.right)/2; cy=(rect.top+rect.bottom)/2
    root=Entity(position=Vec3(cx*SCALE,0,cy*SCALE))
    w=rect.width*SCALE; d=rect.height*SCALE
    FRAME=color.rgba32(90,60,30); SHEET=color.rgba32(230,228,240); PILLOW=color.rgba32(248,245,255)
    # Bed frame
    _box(root,Vec3(0,0.12,0),(w,0.24,d),FRAME)
    # Mattress
    _box(root,Vec3(0,0.26,0),(w*0.96,0.14,d*0.96),color.rgba32(245,240,230))
    # Fitted sheet
    _box(root,Vec3(0,0.34,d*0.10),(w*0.94,0.04,d*0.75),SHEET)
    # Duvet/comforter — slightly raised, textured with seam lines
    _box(root,Vec3(0,0.40,d*0.10),(w*0.93,0.09,d*0.72),color.rgba32(140,160,210))
    # Duvet seam lines
    for sx2 in [-w*0.25,0,w*0.25]:
        _box(root,Vec3(sx2,0.455,d*0.10),(0.012,0.005,d*0.70),color.rgba32(120,140,190))
    # Pillows (2) — at headboard end (+Z)
    for px2 in [-w*0.22,w*0.22]:
        _box(root,Vec3(px2,0.41,d*0.34),(w*0.38,0.08,d*0.26),PILLOW)
    # Headboard — against the far wall (+Z = large sim-Y side)
    _box(root,Vec3(0,0.55,d*0.47),(w,0.50,0.09),FRAME)
    # Headboard panel detail
    _box(root,Vec3(0,0.55,d*0.47),(w*0.88,0.38,0.05),color.rgba32(110,75,38))
    # Legs (4)
    for lx,lz in [(w*0.46,d*0.45),(w*0.46,-d*0.45),(-w*0.46,d*0.45),(-w*0.46,-d*0.45)]:
        _box(root,Vec3(lx,0.04,lz),(0.06,0.08,0.06),color.rgba32(60,38,18))
    # Bedside table — beside headboard (+Z side)
    _box(root,Vec3(w*0.58,0.22,d*0.25),(0.14,0.24,0.14),FRAME)
    _box(root,Vec3(w*0.58,0.36,d*0.25),(0.15,0.03,0.15),color.rgba32(110,75,38))
    _cyl(root,Vec3(w*0.58,0.39,d*0.25-0.04),(0.05,0.09,0.05),color.rgba32(240,220,140),8)
    _cyl(root,Vec3(w*0.58,0.49,d*0.25-0.04),(0.09,0.06,0.09),color.rgba32(230,215,180,200),8)

# ── Obstacle 8: Wardrobe ─────────────────────
def make_wardrobe(rect):
    cx=(rect.left+rect.right)/2; cy=(rect.top+rect.bottom)/2
    # Wardrobe against right wall — rotate so doors open toward -X (into room)
    root=Entity(position=Vec3(cx*SCALE,0,cy*SCALE), rotation_y=90)
    # rect.width=50 (X, becomes depth), rect.height=140 (Z, becomes width along wall)
    w=rect.width*SCALE; d=rect.height*SCALE; H=WALL_H*0.97
    BODY=color.rgba32(200,195,185); TRIM=color.rgba32(230,225,215); HANDLE=color.rgba32(160,145,120)
    # Main body (d = along-wall width, w = depth)
    _box(root,Vec3(0,H/2,0),(d,H,w),BODY)
    # Door panels (2 doors) — face local -Z = world -X = into room
    for dx2 in [-d*0.26,d*0.26]:
        _box(root,Vec3(dx2,H/2,-w*0.52),(d*0.46,H*0.96,0.02),TRIM)
        _box(root,Vec3(dx2,H*0.65,-w*0.535),(d*0.38,H*0.32,0.01),color.rgba32(215,210,200))
        _box(root,Vec3(dx2,H*0.27,-w*0.535),(d*0.38,H*0.38,0.01),color.rgba32(215,210,200))
        _box(root,Vec3(dx2*0.55,H*0.48,-w*0.545),(0.012,0.07,0.012),HANDLE)
    # Top crown moulding
    _box(root,Vec3(0,H+0.02,0),(d*1.02,0.04,w*1.02),TRIM)
    # Bottom plinth
    _box(root,Vec3(0,0.025,0),(d*1.01,0.05,w*1.01),TRIM)

# ── Obstacle 9: Desk + Chair ─────────────────
def make_desk(rect):
    cx=(rect.left+rect.right)/2; cy=(rect.top+rect.bottom)/2
    root=Entity(position=Vec3(cx*SCALE,0,cy*SCALE))
    w=rect.width*SCALE; d=rect.height*SCALE
    OAK=color.rgba32(160,110,55); METAL=color.rgba32(65,65,70); BLACK=color.rgba32(20,20,22)
    # Desk surface
    _box(root,Vec3(0,0.38,0),(w,0.04,d),OAK)
    # Legs (4 metal)
    for lx,lz in [(w*0.44,d*0.40),(w*0.44,-d*0.40),(-w*0.44,d*0.40),(-w*0.44,-d*0.40)]:
        _box(root,Vec3(lx,0.19,lz),(0.03,0.38,0.03),METAL)
    # Monitor
    _box(root,Vec3(0,0.62,d*-0.25),(w*0.55,0.29,0.025),BLACK)          # screen
    _box(root,Vec3(0,0.42,d*-0.25),(0.04,0.08,0.04),METAL)             # stand post
    _box(root,Vec3(0,0.40,d*-0.20),(0.15,0.015,0.09),METAL)            # stand base
    _box(root,Vec3(0,0.635,d*-0.25),(w*0.50,0.25,0.01),color.rgba32(20,30,70))  # display
    # Keyboard
    _box(root,Vec3(0,0.392,d*0.10),(w*0.45,0.015,d*0.22),color.rgba32(35,35,38))
    # Mouse
    _box(root,Vec3(w*0.28,0.392,d*0.12),(0.04,0.018,0.06),color.rgba32(30,30,32))
    # Desk lamp
    _box(root,Vec3(-w*0.36,0.38,d*-0.28),(0.025,0.025,0.025),METAL)
    _box(root,Vec3(-w*0.36,0.55,d*-0.28),(0.015,0.34,0.015),METAL)
    _box(root,Vec3(-w*0.30,0.62,d*-0.28),(0.12,0.015,0.06),METAL)
    # Desk chair
    _box(root,Vec3(0,0.22,d*0.55),(0.20,0.06,0.20),color.rgba32(30,30,30))  # seat
    _box(root,Vec3(0,0.38,d*0.45),(0.20,0.26,0.04),color.rgba32(25,25,25))  # back
    _cyl(root,Vec3(0,0.10,d*0.55),(0.025,0.20,0.025),METAL,6)               # post
    for ax,az in [(0.09,0),(0,0.09),(-0.09,0),(0,-0.09)]:
        _box(root,Vec3(ax,0.02,d*0.55+az),(0.12,0.018,0.018),METAL)         # base arms

# ── BUILD FURNITURE ───────────────────────────
make_tv_unit(obstacles_rects[0])
make_sofa(obstacles_rects[1])
make_coffee_table(obstacles_rects[2])
make_bookshelf(obstacles_rects[3])
make_armchair(obstacles_rects[4])
make_dining_table(obstacles_rects[5])
make_kitchen_counter(obstacles_rects[6])
make_bed(obstacles_rects[7])
make_wardrobe(obstacles_rects[8])
make_desk(obstacles_rects[9])

# Room labels
for lname,lx,ly in [("Living Room",265,350),("Dining Room",715,150),("Bedroom",715,490)]:
    Text(lname,position=p2w(lx,ly)+Vec3(0,0.08,0),
         scale=4.0,color=color.rgba32(255,255,255,200),billboard=True)

# ═════════════════════════════════════════════
# DOG ROBOT  (Buddy)
# A quadruped: body, 4 legs, head, snout, ears, tail, collar
# ═════════════════════════════════════════════
DOG_BODY_COL  = color.rgba32(195,175,145)   # golden retriever sand
DOG_DARK      = color.rgba32(140,110,70)    # darker fur
DOG_NOSE      = color.rgba32(30,25,22)
DOG_EYE       = color.rgba32(55,35,15)
DOG_TONGUE    = color.rgba32(220,80,90)
DOG_COLLAR    = color.rgba32(50,100,200)    # blue tech collar with LED
DOG_COLLAR_LED= color.rgba32(0,220,255)

def make_dog():
    root=Entity()
    S=SCALE*15   # scale reference = robot_radius in world units

    # ── Body ─────────────────────────────────
    body=_box(root,Vec3(0,S*1.1,0),(S*2.6,S*0.85,S*1.3),DOG_BODY_COL)
    # Belly (slightly lighter)
    _box(root,Vec3(0,S*0.78,0),(S*2.2,S*0.30,S*1.0),color.rgba32(220,205,175))
    # Back / topline (darker saddle)
    _box(root,Vec3(0,S*1.50,0),(S*2.4,S*0.18,S*1.1),DOG_DARK)

    # ── Neck & Head ──────────────────────────
    neck=_box(root,Vec3(S*1.1,S*1.55,0),(S*0.55,S*0.70,S*0.60),DOG_BODY_COL)
    # Head block
    head=_box(root,Vec3(S*1.60,S*1.90,0),(S*0.80,S*0.70,S*0.75),DOG_BODY_COL)
    # Snout / muzzle
    snout=_box(root,Vec3(S*2.02,S*1.72,0),(S*0.50,S*0.38,S*0.50),color.rgba32(210,188,155))
    # Nose
    _box(root,Vec3(S*2.28,S*1.80,0),(S*0.18,S*0.16,S*0.22),DOG_NOSE)
    # Nostrils
    _box(root,Vec3(S*2.30,S*1.74, S*0.07),(S*0.04,S*0.05,S*0.07),color.rgba32(15,12,10))
    _box(root,Vec3(S*2.30,S*1.74,-S*0.07),(S*0.04,S*0.05,S*0.07),color.rgba32(15,12,10))
    # Tongue (hangs out slightly)
    _box(root,Vec3(S*2.26,S*1.57, S*0.0),(S*0.10,S*0.14,S*0.18),DOG_TONGUE)
    # Eyes
    _box(root,Vec3(S*2.0,S*2.0, S*0.28),(S*0.10,S*0.12,S*0.10),DOG_EYE)
    _box(root,Vec3(S*2.0,S*2.0,-S*0.28),(S*0.10,S*0.12,S*0.10),DOG_EYE)
    # Eye shine
    _box(root,Vec3(S*2.02,S*2.04, S*0.30),(S*0.04,S*0.04,S*0.04),color.rgba32(240,240,255))
    _box(root,Vec3(S*2.02,S*2.04,-S*0.30),(S*0.04,S*0.04,S*0.04),color.rgba32(240,240,255))
    # Ears (floppy, darker)
    _box(root,Vec3(S*1.62,S*2.10, S*0.45),(S*0.28,S*0.50,S*0.22),DOG_DARK)
    _box(root,Vec3(S*1.62,S*2.10,-S*0.45),(S*0.28,S*0.50,S*0.22),DOG_DARK)

    # ── Legs (4) ─────────────────────────────
    # Front legs
    for lz in [S*0.42,-S*0.42]:
        _box(root,Vec3( S*0.95,S*0.52,lz),(S*0.28,S*0.85,S*0.28),DOG_BODY_COL)  # upper
        _box(root,Vec3( S*0.95,S*0.10,lz),(S*0.22,S*0.50,S*0.26),DOG_DARK)      # lower
        _box(root,Vec3( S*0.95,-S*0.02,lz),(S*0.28,S*0.06,S*0.35),DOG_NOSE)     # paw
    # Rear legs
    for lz in [S*0.42,-S*0.42]:
        _box(root,Vec3(-S*0.90,S*0.58,lz),(S*0.30,S*0.75,S*0.30),DOG_BODY_COL)
        _box(root,Vec3(-S*0.80,S*0.12,lz),(S*0.25,S*0.52,S*0.27),DOG_DARK)
        _box(root,Vec3(-S*0.80,-S*0.02,lz),(S*0.30,S*0.06,S*0.38),DOG_NOSE)

    # ── Tail ─────────────────────────────────
    tail_root=Entity(parent=root,position=Vec3(-S*1.30,S*1.35,0))
    _box(tail_root,Vec3(-S*0.15,S*0.35,0),(S*0.20,S*0.65,S*0.18),DOG_BODY_COL)
    _box(tail_root,Vec3(-S*0.38,S*0.75,0),(S*0.16,S*0.40,S*0.14),DOG_DARK)

    # ── Tech Collar ──────────────────────────
    collar=Entity(parent=root,
                  model=Cylinder(resolution=16,height=1),
                  position=Vec3(S*1.1,S*1.30,0),
                  scale=(S*0.68,S*0.18,S*0.68),
                  color=DOG_COLLAR)
    # LED strip on collar
    _cyl(root,Vec3(S*1.1,S*1.32,0),(S*0.70,S*0.08,S*0.70),DOG_COLLAR_LED,16)

    # FOV cone parent will be attached to root later
    return root, tail_root

dog_ent, dog_tail = make_dog()

# ═════════════════════════════════════════════
# HUMAN FIGURE builder
# Torso, head, neck, arms (upper+lower), legs (upper+lower), feet, hair
# ═════════════════════════════════════════════
def make_human(shirt_col, pants_col, hair_col, skin_col=None):
    if skin_col is None: skin_col=color.rgba32(225,185,140)
    root=Entity()
    S=SCALE*15

    # Legs
    for lz in [S*0.18,-S*0.18]:
        _box(root,Vec3(0,S*0.72,lz),(S*0.30,S*1.10,S*0.28),pants_col)   # upper leg
        _box(root,Vec3(0,S*0.15,lz),(S*0.25,S*0.55,S*0.24),pants_col)   # lower leg
        # Shoe
        _box(root,Vec3(S*0.06,S*-0.02,lz),(S*0.30,S*0.08,S*0.35),color.rgba32(35,30,28))

    # Pelvis / hips
    _box(root,Vec3(0,S*1.32,0),(S*0.62,S*0.25,S*0.52),pants_col)

    # Torso
    _box(root,Vec3(0,S*1.80,0),(S*0.68,S*0.72,S*0.46),shirt_col)
    # Collar (slightly lighter than shirt)
    _box(root,Vec3(0,S*2.18,0),(S*0.38,S*0.10,S*0.34),color.rgba32(
        min(255,int(shirt_col.r*255)+30),
        min(255,int(shirt_col.g*255)+30),
        min(255,int(shirt_col.b*255)+30)))

    # Arms
    for sign,side in [(1,S*0.46),(-1,-S*0.46)]:
        # Upper arm
        _box(root,Vec3(0,S*1.88,side),(S*0.28,S*0.55,S*0.26),shirt_col)
        # Elbow bump
        _box(root,Vec3(S*0.04,S*1.60,side),(S*0.24,S*0.14,S*0.24),skin_col)
        # Forearm
        _box(root,Vec3(S*0.06,S*1.38,side),(S*0.22,S*0.48,S*0.22),skin_col)
        # Hand
        _box(root,Vec3(S*0.08,S*1.10,side),(S*0.20,S*0.22,S*0.20),skin_col)

    # Neck
    _box(root,Vec3(0,S*2.30,0),(S*0.24,S*0.25,S*0.22),skin_col)
    # Head
    head=_box(root,Vec3(0,S*2.65,0),(S*0.50,S*0.58,S*0.46),skin_col)
    # Ears
    _box(root,Vec3(0,S*2.65, S*0.26),(S*0.08,S*0.18,S*0.08),skin_col)
    _box(root,Vec3(0,S*2.65,-S*0.26),(S*0.08,S*0.18,S*0.08),skin_col)
    # Hair
    _box(root,Vec3(0,S*2.92,0),(S*0.52,S*0.22,S*0.48),hair_col)
    _box(root,Vec3(-S*0.08,S*2.78,-S*0.22),(S*0.40,S*0.12,S*0.14),hair_col)
    # Face features
    # Eyes
    _box(root,Vec3(S*0.28,S*2.72, S*0.14),(S*0.06,S*0.07,S*0.07),color.rgba32(40,30,20))
    _box(root,Vec3(S*0.28,S*2.72,-S*0.14),(S*0.06,S*0.07,S*0.07),color.rgba32(40,30,20))
    # Eyebrows
    _box(root,Vec3(S*0.26,S*2.80, S*0.14),(S*0.07,S*0.03,S*0.11),hair_col)
    _box(root,Vec3(S*0.26,S*2.80,-S*0.14),(S*0.07,S*0.03,S*0.11),hair_col)
    # Nose
    _box(root,Vec3(S*0.27,S*2.63,0),(S*0.07,S*0.06,S*0.05),color.rgba32(200,160,120))
    # Mouth
    _box(root,Vec3(S*0.27,S*2.54, S*0.07),(S*0.06,S*0.04,S*0.10),color.rgba32(180,100,90))
    _box(root,Vec3(S*0.27,S*2.54,-S*0.07),(S*0.06,S*0.04,S*0.10),color.rgba32(180,100,90))

    return root, head

# Primary human — blue shirt, grey jeans, brown hair
human_ent,  human_head  = make_human(
    color.rgba32(60,100,175), color.rgba32(80,85,95),  color.rgba32(90,55,20))

# Secondary human — green top, dark pants, black hair
human2_ent, human2_head = make_human(
    color.rgba32(55,135,70),  color.rgba32(45,42,50),  color.rgba32(20,18,16),
    skin_col=color.rgba32(200,155,110))

# ═════════════════════════════════════════════
# DOG HUD OVERLAYS
# ═════════════════════════════════════════════
# Status ring under dog
ring=Entity(model=Circle(resolution=32,mode='line'),
            scale=SCALE*robot_radius*5.5,
            position=Vec3(0,0.005,0),rotation_x=90,
            color=color.rgba32(0,165,200,170))
ring.parent=dog_ent

# FOV cone — filled wedge (more visible)
FOV_SEGS=24
fov_verts=[(0,0,0)]
for i in range(FOV_SEGS+1):
    a=-FOV_HALF+i*(FOV_ANGLE/FOV_SEGS)
    fov_verts.append((math.cos(a)*FOV_RADIUS*SCALE,0,math.sin(a)*FOV_RADIUS*SCALE))
fov_tris=[]
for i in range(1,FOV_SEGS+1):
    fov_tris+=[0,i,i+1 if i<FOV_SEGS else 1]
from ursina.mesh import Mesh as UMesh
fov_mesh=UMesh(vertices=fov_verts,triangles=fov_tris,mode='triangle')
fov_ent=Entity(model=fov_mesh,color=color.rgba32(100,200,255,36),position=Vec3(0,0.006,0))
fov_ent.parent=dog_ent

# FOV arc edge — outline along the cone boundary for crispness
arc_verts=[]
# Left ray
arc_verts.append((0,0,0))
arc_verts.append((math.cos(-FOV_HALF)*FOV_RADIUS*SCALE,0,math.sin(-FOV_HALF)*FOV_RADIUS*SCALE))
# Arc top
for i in range(FOV_SEGS+1):
    a=-FOV_HALF+i*(FOV_ANGLE/FOV_SEGS)
    arc_verts.append((math.cos(a)*FOV_RADIUS*SCALE,0,math.sin(a)*FOV_RADIUS*SCALE))
# Right ray back to origin
arc_verts.append((0,0,0))
fov_edge_mesh=UMesh(vertices=arc_verts,mode='line')
fov_edge=Entity(model=fov_edge_mesh,color=color.rgba32(120,210,255,90),position=Vec3(0,0.007,0))
fov_edge.parent=dog_ent

# Path trail
trail_ent=Entity(model=Mesh(vertices=[],mode='line'),color=color.rgba32(60,160,90,180))

# Audio-guess marker (glowing sphere)
guess_marker=Entity(model='sphere',scale=0.14,
                    color=color.rgba32(210,190,50,220),enabled=False)

# Signal pulse ring
pulse_ent=Entity(model=Circle(resolution=32,mode='line'),
                 scale=0,position=Vec3(0,0.01,0),rotation_x=90,
                 color=color.rgba32(210,88,50,150),enabled=False)
pulse_ent.parent=dog_ent

# Task label above primary human
task_text=Text("",scale=3,color=color.rgba32(220,220,180,220),billboard=True,enabled=True)
task_text.parent=human_ent
S_ref=SCALE*15
task_text.position=Vec3(0,S_ref*3.5,0)

# ── Side Panel HUD ───────────────────────────
# Semi-transparent background panel on the left — narrow strip
panel_bg = Entity(
    model='quad',
    parent=camera.ui,
    position=(-0.815, 0.0, 0),
    scale=(0.34, 1.92),
    color=color.rgba32(8, 10, 16, 155),
)

# Panel title
panel_title = Text(
    "🐕 BUDDY HUD",
    parent=camera.ui,
    position=(-0.815, 0.44),
    scale=1.10,
    color=color.rgba32(190, 200, 220, 210),
)

# Divider line
panel_div = Entity(
    model='quad',
    parent=camera.ui,
    position=(-0.815, 0.415),
    scale=(0.32, 0.002),
    color=color.rgba32(90, 110, 150, 140),
)

# Mode / state heading
state_text = Text(
    "",
    parent=camera.ui,
    position=(-0.815, 0.385),
    scale=1.20,
    color=color.lime,
)

# Sub-info line (attention, scanning, etc.)
info_text = Text(
    "",
    parent=camera.ui,
    position=(-0.815, 0.355),
    scale=0.95,
    color=color.rgba32(185, 185, 195, 195),
)

# Second divider
panel_div2 = Entity(
    model='quad',
    parent=camera.ui,
    position=(-0.815, 0.330),
    scale=(0.32, 0.002),
    color=color.rgba32(70, 90, 130, 110),
)

# Controls heading
ctrl_heading = Text(
    "CONTROLS",
    parent=camera.ui,
    position=(-0.815, 0.305),
    scale=0.95,
    color=color.rgba32(150, 165, 200, 200),
)

_ctrl_lines = [
    ("RMB",     "lock/unlock mouse"),
    ("WASD",    "fly camera"),
    ("Q / E",   "up / down"),
    ("Scroll",  "zoom FOV"),
    ("F",       "first-person"),
    ("R",       "reset camera"),
    ("ESC",     "quit"),
]
_ctrl_y = 0.278
for _key, _desc in _ctrl_lines:
    Text(f"[orange]{_key}[/]  {_desc}",
         parent=camera.ui,
         position=(-0.815, _ctrl_y),
         scale=0.88,
         color=color.rgba32(175, 180, 190, 185))
    _ctrl_y -= 0.034

# Follow-mode indicator near bottom of panel
camera_text = Text(
    "",
    parent=camera.ui,
    position=(-0.815, -0.43),
    scale=0.92,
    color=color.rgba32(165, 175, 190, 185),
)

# ─────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────
camera.orthographic=False; camera.fov=75
FREE_FLY=True
DEFAULT_CAM_POS=Vec3(SIM_W*SCALE/2,7.0,-3.5)
DEFAULT_CAM_ROT=Vec3(52,0,0)
camera.position=DEFAULT_CAM_POS; camera.rotation=DEFAULT_CAM_ROT
_cam_yaw=0.0; _cam_pitch=52.0; _cam_speed=8.0; _mouse_sens=60.0
mouse.locked=False

def reset_camera():
    global _cam_yaw,_cam_pitch
    camera.position=Vec3(DEFAULT_CAM_POS)
    _cam_yaw=0.0; _cam_pitch=52.0
    camera.rotation=Vec3(_cam_pitch,_cam_yaw,0)
    mouse.locked=False

# ─────────────────────────────────────────────
# SIMULATION STATE
# ─────────────────────────────────────────────
_lb=rooms["living"]["bounds"]
rx,ry=safe_spawn(_lb,robot_radius)
robot_dir_x,robot_dir_y=1.0,0.0
robot_angle_cache=0.0; prev_dir_x=1.0; prev_dir_y=0.0
_dog_visual_angle=0.0

current_speed=0.0
prev_distance_error=0.0; integral_error=0.0

robot_state="wander"
search_timer=0.0; signal_timer=0.0; signal_pulse_radius=0.0
waiting_recall_timer=0.0; waiting_recall_duration=0.0
scan_burst_timer=0.0; scan_burst_active=False
ack_active=False; ack_timer=0.0; prev_visible=False
lost_grace_timer=0.0
peek_active=False; peek_goal_x=0.0; peek_goal_y=0.0; peek_done=False
audio_guess_x=0.0; audio_guess_y=0.0; audio_guess_angle=0.0
prev_follow_mode=False
last_seen_x=0.0; last_seen_y=0.0; has_last_seen=False
attention=1.0
human_last_x=rx; human_last_y=ry; human_still_timer=0.0
smoothed_human_dir_x=0.0; smoothed_human_dir_y=0.0

robot_nav_path=[]; robot_nav_wp=0; robot_nav_goal=None
robot_nav_replan_timer=0.0
ROBOT_NAV_REPLAN_INTERVAL=0.4

wander_target_x=200.0; wander_target_y=300.0
wander_timer=0.0; wander_change_time=random.uniform(2,4)

human_room=random.choice(room_order)
tx,ty=safe_spawn(rooms[human_room]["bounds"],target_radius)
target_angle=random.uniform(0,math.pi*2)
target_speed=target_max_speed*random.uniform(0.6,1.0)
human_task_name,_ftp,human_task_dwell=pick_task(human_room)
human_waypoints=[_ftp]; human_wp_index=0
human_task_phase="moving_to_waypoint"; human_task_timer=0.0
human_next_room=None

follow_mode=True
human_toggle_timer=0.0; human_toggle_duration=random.uniform(4,10)
human_calling_pause=False; human_calling_pause_timer=0.0
human_calling_pause_duration=0.0; human_base_speed=target_speed

_h2r=random.choice(["dining","bedroom"])
tx2,ty2=safe_spawn(rooms[_h2r]["bounds"],target_radius)
target2_angle=random.uniform(0,math.pi*2)
target2_speed=target_max_speed*random.uniform(0.4,0.7)

path_trail=[]
_anim_t=0.0   # global time for leg/tail animation

# ─────────────────────────────────────────────
# HUMAN AI MOVEMENT
# ─────────────────────────────────────────────
def move_human_free(hx,hy,h_angle,h_speed,ox,oy):
    # Gentle organic drift — tiny angular nudge scaled to dt (per second, not per frame)
    h_angle += random.uniform(-math.radians(8), math.radians(8)) * dt
    # Soft push away from the other human
    rdx=hx-ox; rdy=hy-oy; rd=math.hypot(rdx,rdy)
    if rd < 90 and rd != 0:
        push = math.atan2(rdy, rdx)
        diff = (push - h_angle + math.pi) % (2*math.pi) - math.pi
        h_angle += diff * 0.08 * dt * (90 - rd) / 90
    # Smooth obstacle avoidance — blend deflect angle gently
    sx,sy=0.0,0.0; avoid_r=45
    for ob in all_collidables:
        cx2=max(ob.left,min(hx,ob.right)); cy2=max(ob.top,min(hy,ob.bottom))
        odx=hx-cx2; ody=hy-cy2; od=math.hypot(odx,ody)
        if od < avoid_r and od != 0:
            t=(avoid_r-od)/avoid_r
            sx+=(odx/od)*t; sy+=(ody/od)*t
    if math.hypot(sx,sy)>0.01:
        sa=math.atan2(sy,sx)
        diff=(sa-h_angle+math.pi)%(2*math.pi)-math.pi
        h_angle+=diff*0.10
    nhx=hx+h_speed*math.cos(h_angle)*dt; nhy=hy+h_speed*math.sin(h_angle)*dt
    if not any(circle_rect_col(nhx,nhy,target_radius,o) for o in all_collidables): hx,hy=nhx,nhy
    elif not any(circle_rect_col(nhx,hy,target_radius,o) for o in all_collidables): hx=nhx
    elif not any(circle_rect_col(hx,nhy,target_radius,o) for o in all_collidables): hy=nhy
    else: h_angle += math.pi*0.5   # clean 90 deg turn on full block, no random snap
    hx=max(target_radius+WALL_THICK,min(SIM_W-target_radius-WALL_THICK,hx))
    hy=max(target_radius+WALL_THICK,min(SIM_H-target_radius-WALL_THICK,hy))
    return hx,hy,h_angle

def move_human_task_fn(hx,hy,h_angle,h_speed,rob_x,rob_y):
    global human_task_phase,human_task_name,human_task_timer,human_task_dwell
    global human_waypoints,human_wp_index,human_next_room,human_room
    if not human_waypoints: return hx,hy,h_angle
    wx,wy=human_waypoints[human_wp_index]
    dxw=wx-hx; dyw=wy-hy; dw=math.hypot(dxw,dyw)
    if dw<14:
        human_wp_index+=1
        if human_wp_index>=len(human_waypoints):
            human_wp_index=0; human_waypoints=[]
            if human_task_phase=="moving_to_door":      human_task_phase="crossing_door"
            elif human_task_phase=="crossing_door":
                human_room=human_next_room
                human_task_name,tp,human_task_dwell=pick_task(human_room)
                human_waypoints=[tp]; human_task_phase="moving_to_waypoint"
            elif human_task_phase in("moving_to_waypoint","at_task"):
                human_task_phase="at_task"; human_task_timer=0.0
        return hx,hy,h_angle
    # Smoothly steer toward waypoint — clamp turn rate so no snapping
    desired = math.atan2(dyw, dxw)
    da = (desired - h_angle + math.pi) % (2*math.pi) - math.pi
    max_turn = math.radians(120) * dt   # max degrees/s
    h_angle += max(-max_turn, min(max_turn, da * 4.0))
    # Speed: slow when turning sharply, slow when very close
    turn_factor = 1.0 - min(abs(da) / math.pi, 1.0) * 0.55
    dist_factor = min(dw / 40.0, 1.0)
    speed = h_speed * turn_factor * dist_factor
    # Gentle obstacle avoidance — nudge angle only
    sx,sy=0.0,0.0; ar=38
    for ob in all_collidables:
        cx2=max(ob.left,min(hx,ob.right)); cy2=max(ob.top,min(hy,ob.bottom))
        odx=hx-cx2; ody=hy-cy2; od=math.hypot(odx,ody)
        if od<ar and od!=0:
            t=(ar-od)/ar
            sx+=(odx/od)*t; sy+=(ody/od)*t
    if math.hypot(sx,sy)>0.01:
        sa=math.atan2(sy,sx)
        diff=(sa-h_angle+math.pi)%(2*math.pi)-math.pi
        h_angle+=diff*0.12
    nhx=hx+speed*math.cos(h_angle)*dt; nhy=hy+speed*math.sin(h_angle)*dt
    if not any(circle_rect_col(nhx,nhy,target_radius,o) for o in all_collidables): hx,hy=nhx,nhy
    elif not any(circle_rect_col(nhx,hy,target_radius,o) for o in all_collidables): hx=nhx
    elif not any(circle_rect_col(hx,nhy,target_radius,o) for o in all_collidables): hy=nhy
    else: h_angle += math.pi*0.5
    hx=max(target_radius+WALL_THICK,min(SIM_W-target_radius-WALL_THICK,hx))
    hy=max(target_radius+WALL_THICK,min(SIM_H-target_radius-WALL_THICK,hy))
    return hx,hy,h_angle

def target_in_fov(robot_angle,txx,tyy):
    dx=txx-rx; dy=tyy-ry
    if dx*dx+dy*dy>FOV_RADIUS*FOV_RADIUS: return False
    ad=abs((math.atan2(dy,dx)-robot_angle+math.pi)%(2*math.pi)-math.pi)
    return ad<FOV_HALF

def target_in_fov_soft(robot_angle,txx,tyy):
    dx=txx-rx; dy=tyy-ry
    if dx*dx+dy*dy>FOV_RADIUS*FOV_RADIUS: return False
    ad=abs((math.atan2(dy,dx)-robot_angle+math.pi)%(2*math.pi)-math.pi)
    return ad<FOV_HALF_SOFT

def nav_dir_to_goal(goal_wx,goal_wy,replan_always=False):
    global robot_nav_path,robot_nav_wp,robot_nav_goal,robot_nav_replan_timer
    LOOKAHEAD=70.0; goal=(goal_wx,goal_wy)
    gm=robot_nav_goal is None or math.hypot(goal[0]-robot_nav_goal[0],goal[1]-robot_nav_goal[1])>35
    robot_nav_replan_timer+=dt
    should=(replan_always and robot_nav_replan_timer>=ROBOT_NAV_REPLAN_INTERVAL) or gm or not robot_nav_path
    if should:
        robot_nav_path=astar((rx,ry),goal); robot_nav_wp=0
        robot_nav_goal=goal; robot_nav_replan_timer=0.0
    if not robot_nav_path:
        ddx=goal_wx-rx; ddy=goal_wy-ry; dm=math.hypot(ddx,ddy) or 0.001
        return ddx/dm,ddy/dm
    while robot_nav_wp<len(robot_nav_path)-1:
        wpx,wpy=robot_nav_path[robot_nav_wp]
        if math.hypot(wpx-rx,wpy-ry)<LOOKAHEAD*0.5: robot_nav_wp+=1
        else: break
    cx2,cy2=rx,ry; rem=LOOKAHEAD
    for i in range(robot_nav_wp,len(robot_nav_path)):
        sx2,sy2=robot_nav_path[i]; sdx=sx2-cx2; sdy=sy2-cy2; sl=math.hypot(sdx,sdy)
        if sl<=0: continue
        if sl>=rem: cx2+=(sdx/sl)*rem; cy2+=(sdy/sl)*rem; rem=0; break
        cx2,cy2=sx2,sy2; rem-=sl
    if rem>0: cx2,cy2=robot_nav_path[-1]
    ddx=cx2-rx; ddy=cy2-ry; dm=math.hypot(ddx,ddy) or 0.001
    return ddx/dm,ddy/dm

# ─────────────────────────────────────────────
# MAIN UPDATE
# ─────────────────────────────────────────────
def update():
    global rx,ry,tx,ty,tx2,ty2
    global target_angle,target2_angle,target_speed,target2_speed
    global robot_dir_x,robot_dir_y,robot_angle_cache,prev_dir_x,prev_dir_y
    global current_speed,prev_distance_error,integral_error
    global robot_state,search_timer,signal_timer,signal_pulse_radius
    global waiting_recall_timer,waiting_recall_duration
    global scan_burst_timer,scan_burst_active
    global ack_active,ack_timer,prev_visible,lost_grace_timer
    global peek_active,peek_goal_x,peek_goal_y,peek_done
    global audio_guess_x,audio_guess_y,audio_guess_angle,prev_follow_mode
    global last_seen_x,last_seen_y,has_last_seen
    global attention,human_last_x,human_last_y,human_still_timer
    global smoothed_human_dir_x,smoothed_human_dir_y
    global follow_mode,human_toggle_timer,human_toggle_duration
    global human_calling_pause,human_calling_pause_timer,human_calling_pause_duration
    global human_base_speed,human_task_phase,human_task_name,human_task_dwell
    global human_waypoints,human_wp_index,human_next_room,human_room
    global human_task_timer
    global wander_target_x,wander_target_y,wander_timer,wander_change_time
    global path_trail,_anim_t
    global _cam_yaw,_cam_pitch,FREE_FLY
    global dt

    dt=time.dt
    _anim_t+=dt

    # ── Camera ──────────────────────────────────
    if mouse.locked:
        _cam_yaw  +=mouse.velocity[0]*_mouse_sens
        _cam_pitch-=mouse.velocity[1]*_mouse_sens
        _cam_pitch =clamp(_cam_pitch,-89,89)
        camera.rotation=Vec3(_cam_pitch,_cam_yaw,0)
    if FREE_FLY and mouse.locked:
        fwd=camera.forward; rt=camera.right; up=Vec3(0,1,0); mv=Vec3(0,0,0)
        if held_keys['w']: mv+=fwd
        if held_keys['s']: mv-=fwd
        if held_keys['a']: mv-=rt
        if held_keys['d']: mv+=rt
        if held_keys['q']: mv-=up
        if held_keys['e']: mv+=up
        if mv.length()>0: camera.position+=mv.normalized()*_cam_speed*time.dt

    # ── Demo toggle ──────────────────────────────
    human_toggle_timer+=dt
    if human_toggle_timer>=human_toggle_duration:
        intended=not follow_mode
        if intended is False and robot_state not in("follow","wander"):
            human_toggle_timer=0; human_toggle_duration=random.uniform(1,3)
        else:
            follow_mode=intended; human_toggle_timer=0
            human_toggle_duration=random.uniform(2,8)
            if intended is True and random.random()<0.70:
                human_calling_pause=True; human_calling_pause_timer=0.0
                human_calling_pause_duration=random.uniform(1.5,3.0)
                target_speed=target_max_speed*0.08

    if human_calling_pause:
        human_calling_pause_timer+=dt
        if human_calling_pause_timer>=human_calling_pause_duration:
            human_calling_pause=False
            target_speed=target_max_speed*random.uniform(0.5,1.0)
            human_base_speed=target_speed

    # ── Human 1 AI ───────────────────────────────
    if human_task_phase=="at_task":
        human_task_timer+=dt
        # Gentle weight-shift: slowly rotate on the spot, no position jitter
        target_angle += math.sin(_anim_t * 0.4) * math.radians(0.8) * dt
        if human_task_timer>=human_task_dwell:
            if random.random()<0.65:
                human_next_room=pick_next_room(human_room)
                pk=(human_room,human_next_room)
                human_waypoints=door_paths.get(pk,[])[:]
                human_wp_index=0; human_task_phase="moving_to_door"
            else:
                human_task_name,tp,human_task_dwell=pick_task(human_room)
                human_waypoints=[tp]; human_wp_index=0
                human_task_phase="moving_to_waypoint"
    elif human_task_phase in("moving_to_waypoint","moving_to_door","crossing_door"):
        ms=target_speed if not human_calling_pause else target_max_speed*0.08
        tx,ty,target_angle=move_human_task_fn(tx,ty,target_angle,ms,rx,ry)

    tx2,ty2,target2_angle=move_human_free(tx2,ty2,target2_angle,target2_speed,rx,ry)

    # ── Waiting recall ────────────────────────────
    if robot_state=="waiting":
        waiting_recall_timer+=dt
        if waiting_recall_timer>=waiting_recall_duration:
            if random.random()<0.70 and not human_calling_pause:
                human_calling_pause=True; human_calling_pause_timer=0.0
                human_calling_pause_duration=random.uniform(1.5,3.0)
                target_speed=target_max_speed*0.08
            true_angle=math.atan2(ty-ry,tx-rx)
            noise_angle=true_angle+random.uniform(-math.radians(25),math.radians(25))
            true_dist=math.hypot(tx-rx,ty-ry)
            noisy_dist=true_dist*random.uniform(0.75,1.1)
            audio_guess_x=max(50,min(SIM_W-50,rx+math.cos(noise_angle)*noisy_dist))
            audio_guess_y=max(50,min(SIM_H-50,ry+math.sin(noise_angle)*noisy_dist))
            audio_guess_x,audio_guess_y=safe_audio_guess(audio_guess_x,audio_guess_y)
            audio_guess_angle=noise_angle
            robot_state="approach"; integral_error=0.0
            scan_burst_active=False; peek_active=False; peek_done=False; robot_nav_goal=None

    # ── Call-out detection ────────────────────────
    just_called=follow_mode and not prev_follow_mode
    prev_follow_mode=follow_mode
    if just_called:
        true_angle=math.atan2(ty-ry,tx-rx)
        noise_angle=true_angle+random.uniform(-math.radians(25),math.radians(25))
        noisy_dist=math.hypot(tx-rx,ty-ry)*random.uniform(0.75,1.1)
        audio_guess_x=max(50,min(SIM_W-50,rx+math.cos(noise_angle)*noisy_dist))
        audio_guess_y=max(50,min(SIM_H-50,ry+math.sin(noise_angle)*noisy_dist))
        audio_guess_x,audio_guess_y=safe_audio_guess(audio_guess_x,audio_guess_y)
        audio_guess_angle=noise_angle; robot_state="approach"
        integral_error=0.0; scan_burst_active=False
        peek_active=False; peek_done=False; robot_nav_goal=None

    # ── Angle cache ───────────────────────────────
    if robot_dir_x!=prev_dir_x or robot_dir_y!=prev_dir_y:
        robot_angle_cache=math.atan2(robot_dir_y,robot_dir_x)
        prev_dir_x=robot_dir_x; prev_dir_y=robot_dir_y

    # ── Visibility ────────────────────────────────
    in_cone     =target_in_fov(robot_angle_cache,tx,ty)
    in_cone_soft=target_in_fov_soft(robot_angle_cache,tx,ty)
    occluded    =ray_hits_obstacle(rx,ry,tx,ty,obstacles_rects)
    visible     =in_cone and not occluded
    soft_visible=in_cone_soft and not occluded

    if robot_state=="follow":
        if not soft_visible: lost_grace_timer+=dt
        else: lost_grace_timer=0.0
    else: lost_grace_timer=0.0
    truly_lost=lost_grace_timer>=LOST_GRACE_TIME

    if visible and scan_burst_active: scan_burst_active=False
    if robot_state=="follow" and truly_lost and not has_last_seen:
        last_seen_x=tx; last_seen_y=ty; has_last_seen=True
    if visible and not prev_visible and robot_state in("approach","search","wander"):
        ack_active=True; ack_timer=0.0
    if ack_active:
        ack_timer+=dt
        if ack_timer>=ACK_DURATION: ack_active=False
    prev_visible=visible

    # ── Attention ─────────────────────────────────
    if robot_state=="follow":
        hm=math.hypot(tx-human_last_x,ty-human_last_y)
        if hm<2.0: human_still_timer+=dt
        else: human_still_timer=0.0; attention=min(1.0,attention+dt*0.4)
        if human_still_timer>ATTENTION_DECAY_TIME:
            attention=max(ATTENTION_MIN,attention-dt*0.12)
    else: attention=min(1.0,attention+dt*0.5)
    human_last_x,human_last_y=tx,ty

    # ── State machine ─────────────────────────────
    if not follow_mode:
        if robot_state not in("wander",):
            robot_state="wander"; wander_timer=0.0
            integral_error=0.0; scan_burst_active=False
    else:
        if robot_state=="approach":
            if visible:
                robot_state="follow"; integral_error=0.0; peek_active=False
            else:
                dg=math.hypot(audio_guess_x-rx,audio_guess_y-ry)
                if peek_active:
                    dp=math.hypot(peek_goal_x-rx,peek_goal_y-ry)
                    if dp<35:
                        peek_active=False; peek_done=True
                        audio_guess_angle=math.atan2(audio_guess_y-ry,audio_guess_x-rx)
                        robot_state="search"; search_timer=0.0
                        scan_burst_active=True; scan_burst_timer=0.0; robot_nav_goal=None
                elif dg<40:
                    if not peek_done and ray_hits_obstacle(rx,ry,audio_guess_x,audio_guess_y,all_collidables):
                        pp=find_peek_position(rx,ry,audio_guess_x,audio_guess_y)
                        if pp:
                            peek_goal_x,peek_goal_y=pp; peek_active=True; robot_nav_goal=None
                        else:
                            robot_state="search"; search_timer=0.0
                            scan_burst_active=True; scan_burst_timer=0.0
                    else:
                        robot_state="search"; search_timer=0.0
                        scan_burst_active=True; scan_burst_timer=0.0
        elif robot_state=="follow":
            if truly_lost:
                if has_last_seen:
                    audio_guess_x,audio_guess_y=safe_audio_guess(last_seen_x,last_seen_y)
                    audio_guess_angle=math.atan2(last_seen_y-ry,last_seen_x-rx)
                    robot_state="approach"; integral_error=0.0
                    scan_burst_active=False; has_last_seen=False
                else:
                    robot_state="search"; search_timer=0.0
                    scan_burst_active=True; scan_burst_timer=0.0
        elif robot_state=="search":
            search_timer+=dt
            if visible: robot_state="follow"; integral_error=0.0; scan_burst_active=False
            elif search_timer>SEARCH_TIMEOUT:
                robot_state="signal"; signal_timer=0.0; signal_pulse_radius=0.0; scan_burst_active=False
        elif robot_state=="signal":
            signal_timer+=dt; signal_pulse_radius=(signal_timer/SIGNAL_DURATION)*80
            if visible: robot_state="follow"; integral_error=0.0
            elif signal_timer>=SIGNAL_DURATION:
                robot_state="waiting"; waiting_recall_timer=0.0
                waiting_recall_duration=random.uniform(1.5,3.5)
        elif robot_state=="waiting":
            if visible: robot_state="follow"; integral_error=0.0
        elif robot_state=="wander":
            if visible: robot_state="follow"; integral_error=0.0

    # ── Obstacle avoidance ────────────────────────
    avoid_x=0.0; avoid_y=0.0; soft_radius=55; soft_weight=1.8
    for ob in all_collidables:
        clx=max(ob.left,min(rx,ob.right)); cly=max(ob.top,min(ry,ob.bottom))
        dox=rx-clx; doy=ry-cly; dd=math.hypot(dox,doy)
        if dd<soft_radius and dd!=0:
            t=(soft_radius-dd)/soft_radius; s=t*t*soft_weight
            avoid_x+=(dox/dd)*s; avoid_y+=(doy/dd)*s
    dx2p=rx-tx2; dy2p=ry-ty2; d2p=math.hypot(dx2p,dy2p)
    if d2p<55 and d2p!=0:
        t2=(55-d2p)/55; avoid_x+=(dx2p/d2p)*t2*t2; avoid_y+=(dy2p/d2p)*t2*t2

    _mwd=float('inf')
    for _wb in wall_rects:
        _cx2=max(_wb.left,min(rx,_wb.right)); _cy2=max(_wb.top,min(ry,_wb.bottom))
        _d=math.hypot(rx-_cx2,ry-_cy2)
        if _d<_mwd: _mwd=_d
    _wp=max(0.0,1.0-_mwd/60.0)

    # ── Per-state direction + speed ───────────────
    dx=tx-rx; dy=ty-ry; distance=math.hypot(dx,dy) or 0.001
    desired_distance=65; distance_error=distance-desired_distance; dir_x=0.0; dir_y=0.0

    if robot_state=="follow":
        # Compute heel position (slightly behind and to the side of human)
        rdx=tx-human_last_x; rdy=ty-human_last_y; rm=math.hypot(rdx,rdy)
        if rm>0.5:
            smoothed_human_dir_x+=(rdx/rm-smoothed_human_dir_x)*HUMAN_DIR_SMOOTH
            smoothed_human_dir_y+=(rdy/rm-smoothed_human_dir_y)*HUMAN_DIR_SMOOTH
        sm=math.hypot(smoothed_human_dir_x,smoothed_human_dir_y)
        if sm>0.15:
            hda=math.atan2(smoothed_human_dir_y,smoothed_human_dir_x)
            ha=hda+math.pi+HEEL_OFFSET_ANGLE
            heel_x=tx+math.cos(ha)*HEEL_OFFSET_DIST
            heel_y=ty+math.sin(ha)*HEEL_OFFSET_DIST
        else:
            heel_x=tx+(rx-tx)/distance*desired_distance
            heel_y=ty+(ry-ty)/distance*desired_distance

        dir_x,dir_y=nav_dir_to_goal(heel_x,heel_y,replan_always=True)
        hdx=heel_x-rx; hdy=heel_y-ry; hdist=math.hypot(hdx,hdy) or 0.001
        distance_error=hdist-desired_distance

        # PID — no derivative boost, clamped integral
        integral_error+=distance_error*dt
        integral_error=max(-150.0,min(150.0,integral_error))
        derivative=(distance_error-prev_distance_error)/dt
        ds=(Kp_distance*distance_error + Ki_distance*integral_error + Kd_distance*derivative)
        ds*=attention
        # Gentle brake when too close
        if hdist<(desired_distance*0.6):
            ds=-50*(1.0-hdist/(desired_distance*0.6))
        ds=max(-60,min(ds,max_speed))
        # Slow lerp for acceleration — feels weighty, not snappy
        current_speed+=(ds-current_speed)*0.08

    elif robot_state=="approach":
        ntx=peek_goal_x if peek_active else audio_guess_x
        nty=peek_goal_y if peek_active else audio_guess_y
        dir_x,dir_y=nav_dir_to_goal(ntx,nty)
        current_speed+=(max_speed*0.70-current_speed)*0.07

    elif robot_state=="search":
        if scan_burst_active:
            scan_burst_timer+=dt
            t_norm=scan_burst_timer/SCAN_BURST_DURATION
            sa=audio_guess_angle+t_norm*math.pi*2
            robot_dir_x=math.cos(sa); robot_dir_y=math.sin(sa)
            if scan_burst_timer>=SCAN_BURST_DURATION: scan_burst_active=False
        else:
            a=math.atan2(robot_dir_y,robot_dir_x)+SEARCH_ROTATE_SPEED*dt
            robot_dir_x=math.cos(a); robot_dir_y=math.sin(a)
        dir_x=robot_dir_x; dir_y=robot_dir_y
        # Decelerate to a slow creep while scanning
        current_speed+=(0-current_speed)*0.08

    elif robot_state=="signal":
        dir_x=robot_dir_x; dir_y=robot_dir_y
        current_speed+=(0-current_speed)*0.10

    elif robot_state=="waiting":
        a=math.atan2(robot_dir_y,robot_dir_x)+SEARCH_ROTATE_SPEED*0.4*dt
        robot_dir_x=math.cos(a); robot_dir_y=math.sin(a)
        dir_x=robot_dir_x; dir_y=robot_dir_y
        current_speed+=(0-current_speed)*0.10

    else:  # wander
        wander_timer+=dt; dw2=math.hypot(wander_target_x-rx,wander_target_y-ry)
        if dw2<50 or wander_timer>wander_change_time:
            wander_target_x=random.randint(WALL_THICK+40,SIM_W-WALL_THICK-40)
            wander_target_y=random.randint(WALL_THICK+40,SIM_H-WALL_THICK-40)
            wander_timer=0.0; wander_change_time=random.uniform(3,7); robot_nav_goal=None
        dir_x,dir_y=nav_dir_to_goal(wander_target_x,wander_target_y)
        current_speed+=(idle_speed-current_speed)*0.04   # very gradual cruise speed

    prev_distance_error=distance_error

    # ── Heading blend — rate-limited, dt-scaled ───
    # Merge nav direction with obstacle avoidance vector
    if not scan_burst_active and robot_state not in("waiting","signal","search"):
        cx2=dir_x+avoid_x*0.7; cy2=dir_y+avoid_y*0.7
        mg=math.hypot(cx2,cy2)
        if mg!=0: cx2/=mg; cy2/=mg
        # Lerp rate: slow near walls (more cautious), normal otherwise
        # dt-scaled so frame-rate independent
        lerp_rate = (0.08 + _wp*0.08) * (dt * 60)
        lerp_rate = min(lerp_rate, 0.25)   # hard cap so it never snaps
        robot_dir_x+=(cx2-robot_dir_x)*lerp_rate
        robot_dir_y+=(cy2-robot_dir_y)*lerp_rate
        hl=math.hypot(robot_dir_x,robot_dir_y)
        if hl!=0: robot_dir_x/=hl; robot_dir_y/=hl

    dsc=max_speed*(1.0-_wp*0.40)
    current_speed=max(-50,min(current_speed,dsc))
    rx+=current_speed*robot_dir_x*dt; ry+=current_speed*robot_dir_y*dt
    rx=max(robot_radius+10,min(SIM_W-robot_radius-10,rx))
    ry=max(robot_radius+10,min(SIM_H-robot_radius-10,ry))

    for ob in all_collidables:
        if circle_rect_col(rx,ry,robot_radius,ob):
            clx=max(ob.left,min(rx,ob.right)); cly=max(ob.top,min(ry,ob.bottom))
            pdx=rx-clx; pdy=ry-cly; pd=math.hypot(pdx,pdy)
            if pd==0: pdx,pdy,pd=0,-1,1
            ov=robot_radius-pd; rx+=(pdx/pd)*(ov+1); ry+=(pdy/pd)*(ov+1)
            current_speed*=0.45

    # ═══════════════════════════════════════════
    # UPDATE 3-D POSITIONS
    # ═══════════════════════════════════════════

    # ── Dog (robot) ──────────────────────────────
    dog_ent.position=Vec3(rx*SCALE,0,ry*SCALE)

    # Smooth visual rotation — rate-limited so the dog physically turns, never snaps
    global _dog_visual_angle
    target_visual_angle = math.atan2(robot_dir_y, robot_dir_x)
    angle_diff = (target_visual_angle - _dog_visual_angle + math.pi) % (2*math.pi) - math.pi
    max_rot = DOG_MAX_TURN_RATE * dt
    _dog_visual_angle += max(-max_rot, min(max_rot, angle_diff))
    dog_ent.rotation_y = -math.degrees(_dog_visual_angle)

    # Tail wag: oscillate tail_root rotation when following or happy
    is_happy=(robot_state in("follow","approach"))
    wag_speed = 6.0 if is_happy else 1.5
    wag_amp   = 35  if is_happy else 8
    dog_tail.rotation_z=math.sin(_anim_t*wag_speed)*wag_amp

    # ── Humans ───────────────────────────────────
    human_ent.position =Vec3(tx*SCALE,0,ty*SCALE)
    human_ent.rotation_y=-math.degrees(target_angle)
    human2_ent.position=Vec3(tx2*SCALE,0,ty2*SCALE)
    human2_ent.rotation_y=-math.degrees(target2_angle)

    # Collar LED colour by state
    state_colors={
        "follow":  color.rgba32(0,  185,210),
        "approach":color.rgba32(65, 185,210),
        "search":  color.rgba32(210,190,  0),
        "signal":  color.rgba32(210, 55, 28),
        "waiting": color.rgba32(210, 88, 18),
        "wander":  color.rgba32(165, 68,210),
    }
    rc=state_colors.get(robot_state,color.white)
    ri,gi,bi=int(rc.r*255),int(rc.g*255),int(rc.b*255)
    # Ring reflects state
    ring.color=color.rgba32(ri,gi,bi,110)
    # FOV cone tinted by state
    fov_ent.color =color.rgba32(ri,gi,bi,36)
    fov_edge.color=color.rgba32(ri,gi,bi,95)

    # Guess marker
    if robot_state=="approach":
        guess_marker.enabled=True
        guess_marker.position=Vec3(audio_guess_x*SCALE,0.18,audio_guess_y*SCALE)
    else:
        guess_marker.enabled=False

    # Signal pulse
    if robot_state=="signal" and signal_pulse_radius>0:
        pulse_ent.enabled=True
        ps=signal_pulse_radius*SCALE*2
        pulse_ent.scale=Vec3(ps,1,ps)
        pa=max(0,int(150*(1-signal_timer/SIGNAL_DURATION)))
        pulse_ent.color=color.rgba32(210,88,50,pa)
    else:
        pulse_ent.enabled=False

    # Task label
    pl2={"at_task":human_task_name,"moving_to_waypoint":f"→ {human_task_name}",
         "moving_to_door":f"→ {human_next_room or ''}","crossing_door":"crossing..."}
    task_text.text=pl2.get(human_task_phase,"")

    # Trail
    path_trail.append((rx,ry))
    if len(path_trail)>MAX_PATH_LEN: path_trail.pop(0)
    if len(path_trail)>2:
        trail_verts=[Vec3(px2*SCALE,0.008,py2*SCALE) for px2,py2 in path_trail]
        trail_ent.model=Mesh(vertices=trail_verts,mode='line')

    # HUD
    sc2={"follow":"#5FD96E","approach":"#7EB8D8","search":"#D4C05A",
         "signal":"#D06858","waiting":"#C87840","wander":"#9B72CC"}.get(robot_state,"#CCCCCC")
    state_text.text=f"[{sc2}]● {robot_state.upper()}[/]"
    info_lines=[]
    if robot_state=="follow" and attention<0.95:
        info_lines.append(f"Attention: {int(attention*100)}%")
    if robot_state=="search":
        info_lines.append("Looking around..." if scan_burst_active
                          else f"Scanning… {max(0,SEARCH_TIMEOUT-search_timer):.1f}s")
    elif robot_state=="approach": info_lines.append("Heading to audio cue…")
    elif robot_state=="signal":   info_lines.append("Signalling: where are you?!")
    elif robot_state=="waiting":  info_lines.append("Waiting… call me!")
    info_text.text=" | ".join(info_lines)
    follow_lbl="[green]FOLLOWING[/]" if follow_mode else "[orange]WANDERING[/]"
    camera_text.text=f"Mode: {follow_lbl}  |  Mouse: {'locked' if mouse.locked else 'free'}"


# ─────────────────────────────────────────────
# KEY HANDLERS
# ─────────────────────────────────────────────
def input(key):
    global FREE_FLY,_cam_yaw,_cam_pitch
    if key=='right mouse down':
        mouse.locked=not mouse.locked
    if key=='escape':
        application.quit()
    if key=='r':
        reset_camera()
    if key=='f':
        FREE_FLY=not FREE_FLY
        if not FREE_FLY:
            camera.position=dog_ent.position+Vec3(0,1.8,-1.5)
            camera.look_at(dog_ent.position+Vec3(0,0.4,0))
            mouse.locked=True
        else:
            reset_camera()
    if key=='scroll up':   camera.fov=max(20,camera.fov-5)
    if key=='scroll down': camera.fov=min(120,camera.fov+5)


# ─────────────────────────────────────────────
# START
# ─────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════╗
║   Buddy Follow-Me — Realistic 3D Simulation      ║
╠══════════════════════════════════════════════════╣
║  RMB          lock / unlock mouse                ║
║  W/A/S/D/Q/E  fly camera (when locked)           ║
║  Mouse        look around (when locked)          ║
║  Scroll       zoom FOV                           ║
║  F            toggle first-person (dog) mode     ║
║  R            reset camera to overview           ║
║  ESC          quit                               ║
╚══════════════════════════════════════════════════╝
""")
app.run()