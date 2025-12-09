import sys
import math
from collections import defaultdict

# ==========================================
# GEOMETRY ENGINE
# ==========================================
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def orientation(p, q, r):
    """Calcule l'orientation du triplet (p, q, r)"""
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2

def segments_intersect(p1, q1, p2, q2):
    """Vérifie si deux segments s'intersectent"""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    if o1 != o2 and o3 != o4:
        return True
    return False

# ==========================================
# DATA STRUCTURES
# ==========================================
class Building:
    def __init__(self, id, type, x, y):
        self.id = id
        self.type = type
        self.x = x
        self.y = y
        self.astronaut_groups = {}
        self.total_astronauts = 0
        self.degree = 0
    
    def distance_to(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx*dx + dy*dy)

class Tube:
    def __init__(self, u, v, capacity):
        self.u = u
        self.v = v
        self.capacity = capacity
    
    def matches(self, id1, id2):
        return (self.u == id1 and self.v == id2) or (self.u == id2 and self.v == id1)

# ==========================================
# GAME STATE
# ==========================================
class GameState:
    def __init__(self):
        self.buildings = {}
        self.tubes = []
        self.routes_with_pods = set()
        self.next_pod_id = 1
        self.month = 0
        self.module_occupancy = {}
    
    def reset_month(self):
        self.module_occupancy.clear()
        self.tubes.clear()
        self.routes_with_pods.clear()
        for b in self.buildings.values():
            b.degree = 0
    
    def distance(self, id1, id2):
        b1 = self.buildings[id1]
        b2 = self.buildings[id2]
        return b1.distance_to(b2)
    
    def is_blocked(self, u, v):
        """Vérifie si un tube croiserait un tube existant"""
        if u not in self.buildings or v not in self.buildings:
            return True
            
        p1 = Point(self.buildings[u].x, self.buildings[u].y)
        q1 = Point(self.buildings[v].x, self.buildings[v].y)
        
        for tube in self.tubes:
            if tube.u in (u, v) or tube.v in (u, v):
                continue
            
            if tube.u not in self.buildings or tube.v not in self.buildings:
                continue
                
            p2 = Point(self.buildings[tube.u].x, self.buildings[tube.u].y)
            q2 = Point(self.buildings[tube.v].x, self.buildings[tube.v].y)
            
            if segments_intersect(p1, q1, p2, q2):
                return True
        
        return False
    
    def find_tube(self, id1, id2):
        """Trouve un tube existant"""
        for tube in self.tubes:
            if tube.matches(id1, id2):
                return tube
        return None
    
    def tube_cost(self, id1, id2):
        """Coût de construction d'un tube"""
        d = self.distance(id1, id2)
        return int(d * 10)
    
    def upgrade_cost(self, id1, id2, current_capacity):
        """Coût d'upgrade"""
        base_cost = self.tube_cost(id1, id2)
        return base_cost * (current_capacity + 1)

# ==========================================
# STRATEGY
# ==========================================
def plan_actions(state, resources):
    """Planifie les actions"""
    actions = []
    
    # Trier les pads par astronautes
    pads = [b for b in state.buildings.values() if b.type == 0]
    pads.sort(key=lambda p: p.total_astronauts, reverse=True)
    
    # Recalculer les degrés
    for tube in state.tubes:
        if tube.u in state.buildings:
            state.buildings[tube.u].degree += 1
        if tube.v in state.buildings:
            state.buildings[tube.v].degree += 1
    
    # Pour chaque pad et groupe d'astronautes
    for pad in pads:
        for atype, count in pad.astronaut_groups.items():
            if atype == 0 or count == 0:
                continue
            
            # Trouver le meilleur module
            best_module_id = None
            best_score = float('inf')
            
            for building in state.buildings.values():
                if building.type == atype:
                    distance = pad.distance_to(building)
                    occupancy_penalty = state.module_occupancy.get(building.id, 0) * 5.0
                    score = distance + occupancy_penalty
                    
                    if score < best_score:
                        existing_tube = state.find_tube(pad.id, building.id)
                        if existing_tube or not state.is_blocked(pad.id, building.id):
                            best_score = score
                            best_module_id = building.id
            
            if best_module_id is None:
                continue
            
            state.module_occupancy[best_module_id] = state.module_occupancy.get(best_module_id, 0) + count
            
            # CONSTRUIRE TUBE
            existing_tube = state.find_tube(pad.id, best_module_id)
            
            if not existing_tube:
                cost = state.tube_cost(pad.id, best_module_id)
                
                if (resources >= cost and 
                    state.buildings[pad.id].degree < 5 and 
                    state.buildings[best_module_id].degree < 5):
                    
                    actions.append(f"TUBE {pad.id} {best_module_id}")
                    resources -= cost
                    
                    new_tube = Tube(pad.id, best_module_id, 1)
                    state.tubes.append(new_tube)
                    state.buildings[pad.id].degree += 1
                    state.buildings[best_module_id].degree += 1
                    existing_tube = new_tube
            
            # UPGRADE
            if existing_tube and count > existing_tube.capacity * 10:
                cost = state.upgrade_cost(existing_tube.u, existing_tube.v, existing_tube.capacity)
                
                if resources >= cost:
                    actions.append(f"UPGRADE {existing_tube.u} {existing_tube.v}")
                    resources -= cost
                    existing_tube.capacity += 1
            
            # CRÉER PODS
            if existing_tube:
                route_key = (min(pad.id, best_module_id), max(pad.id, best_module_id))
                
                if route_key not in state.routes_with_pods and resources >= 1000:
                    actions.append(f"POD {state.next_pod_id} {pad.id} {best_module_id} {pad.id}")
                    resources -= 1000
                    state.next_pod_id += 1
                    state.routes_with_pods.add(route_key)
    
    # TÉLÉPORTEURS (fin de partie)
    if state.month >= 15 and resources >= 5000:
        for pad in pads[:2]:  # Top 2 pads
            for building in list(state.buildings.values())[:5]:  # Top 5 modules
                if building.type > 0:
                    distance = pad.distance_to(building)
                    if distance > 50 and resources >= 5000:
                        actions.append(f"TELEPORT {pad.id} {building.id}")
                        resources -= 5000
                        break
    
    state.month += 1
    return actions if actions else ["WAIT"]

# ==========================================
# MAIN LOOP
# ==========================================
state = GameState()

while True:
    try:
        state.reset_month()
        
        resources = int(input())
        
        # Routes
        num_routes = int(input())
        for _ in range(num_routes):
            u, v, capacity = map(int, input().split())
            state.tubes.append(Tube(u, v, capacity))
        
        # Pods
        num_pods = int(input())
        max_pod_id = 0
        for _ in range(num_pods):
            line = input().split()
            pod_id = int(line[0])
            max_pod_id = max(max_pod_id, pod_id)
            num_stops = int(line[1])
            
            if len(line) >= 2 + num_stops:
                path = [int(line[2 + i]) for i in range(num_stops)]
                
                for i in range(len(path) - 1):
                    route_key = (min(path[i], path[i+1]), max(path[i], path[i+1]))
                    state.routes_with_pods.add(route_key)
        
        state.next_pod_id = max(state.next_pod_id, max_pod_id + 1)
        
        # Nouveaux bâtiments
        num_new = int(input())
        for _ in range(num_new):
            line = input().split()
            
            if len(line) < 4:
                continue
            
            # Landing Pad
            if line[0] == '0':
                if len(line) >= 5:
                    building_id = int(line[1])
                    x, y = int(line[2]), int(line[3])
                    building = Building(building_id, 0, x, y)
                    
                    num_astro_types = int(line[4])
                    idx = 5
                    
                    for _ in range(num_astro_types):
                        if idx + 1 < len(line):
                            atype = int(line[idx])
                            acount = int(line[idx + 1])
                            building.astronaut_groups[atype] = acount
                            building.total_astronauts += acount
                            idx += 2
                    
                    state.buildings[building_id] = building
            
            # Module
            else:
                building_type = int(line[0])
                building_id = int(line[1])
                x, y = int(line[2]), int(line[3])
                building = Building(building_id, building_type, x, y)
                state.buildings[building_id] = building
        
        # Planifier et exécuter
        actions = plan_actions(state, resources)
        print(";".join(actions), flush=True)
        
    except Exception as e:
        print(f"WAIT", flush=True)
        print(f"Error: {e}", file=sys.stderr, flush=True)