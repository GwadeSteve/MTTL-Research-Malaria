#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <map>

using namespace std;

struct Point {
    int x, y;
};

struct Building {
    int id;
    int type;
    int x, y;
    vector<int> astroTypes;
};

struct Route {
    int u, v; 
    int capacity;
    bool operator<(const Route& other) const {
        if (min(u, v) != min(other.u, other.v))
            return min(u, v) < min(other.u, other.v);
        return max(u, v) < max(other.u, other.v);
    }
};

vector<Building> allBuildings;
int nextPodId = 1;

double getDistance(int x1, int y1, int x2, int y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

int calculateTubeCost(double distance) {
    // Rule: 1 resource per 0.1km. 
    // Usually dist is Euclidean. Let's assume floor(dist * 10) based on typical CG rules
    return (int)floor(distance * 10.0); 
}

int main()
{
    while (1) {
        int resources;
        cin >> resources; cin.ignore();
        
        int num_travel_routes;
        cin >> num_travel_routes; cin.ignore();
        
        // Track existing routes to avoid rebuilding
        set<Route> existingRoutes;
        
        for (int i = 0; i < num_travel_routes; i++) {
            int u, v, capacity;
            cin >> u >> v >> capacity; cin.ignore();
            existingRoutes.insert({u, v, capacity});
        }
        
        int num_pods;
        cin >> num_pods; cin.ignore();
        
        // Track which routes have pods
        // We map a "Route" to a boolean (true if a pod traverses it)
        set<Route> routesWithPods;
        
        int maxExistingPodId = 0;

        for (int i = 0; i < num_pods; i++) {
            int podId, numStops;
            cin >> podId >> numStops;
            maxExistingPodId = max(maxExistingPodId, podId);
            
            vector<int> path;
            for (int k = 0; k < numStops; k++) {
                int stopId;
                cin >> stopId;
                path.push_back(stopId);
            }
            cin.ignore();
            
            // Mark routes used by this pod as "Active"
            for (size_t k = 0; k < path.size() - 1; k++) {
                routesWithPods.insert({path[k], path[k+1], 0});
            }
            // Handle loop back if relevant (though pods usually just follow the list)
        }
        
        // Update our Pod ID counter to avoid duplicates
        nextPodId = max(nextPodId, maxExistingPodId + 1);

        int num_new_buildings;
        cin >> num_new_buildings; cin.ignore();
        
        for (int i = 0; i < num_new_buildings; i++) {
            string building_properties;
            getline(cin, building_properties);
            
            // Manual parsing because format varies
            // Landing Pad: 0 id x y numAstro type1 type2...
            // Module: type id x y
            
            Building b;
            size_t p0 = 0, p1 = building_properties.find(' ');
            int firstToken = stoi(building_properties.substr(p0, p1 - p0));
            
            p0 = p1 + 1; p1 = building_properties.find(' ', p0);
            int secondToken = stoi(building_properties.substr(p0, p1 - p0));
            
            p0 = p1 + 1; p1 = building_properties.find(' ', p0);
            int thirdToken = stoi(building_properties.substr(p0, p1 - p0));
            
            p0 = p1 + 1; p1 = building_properties.find(' ', p0);
            int fourthToken = (p1 == string::npos) ? stoi(building_properties.substr(p0)) : stoi(building_properties.substr(p0, p1 - p0));

            if (firstToken == 0) {
                // LANDING PAD
                b.type = 0;
                b.id = secondToken;
                b.x = thirdToken;
                b.y = fourthToken;
                
                // Parse astronauts
                if (p1 != string::npos) {
                    p0 = p1 + 1; p1 = building_properties.find(' ', p0);
                    int numAstro = stoi(building_properties.substr(p0, p1 - p0));
                    
                    for(int k=0; k<numAstro; k++) {
                        p0 = p1 + 1; p1 = building_properties.find(' ', p0);
                        string sType = (p1 == string::npos) ? building_properties.substr(p0) : building_properties.substr(p0, p1 - p0);
                        b.astroTypes.push_back(stoi(sType));
                    }
                }
            } else {
                // MODULE
                b.type = firstToken;
                b.id = secondToken;
                b.x = thirdToken;
                b.y = fourthToken;
            }
            allBuildings.push_back(b);
        }

        vector<string> actions;
        
        // 1. Identify Pads and Modules
        vector<Building*> pads;
        vector<Building*> modules;
        
        for (auto &b : allBuildings) {
            if (b.type == 0) pads.push_back(&b);
            else modules.push_back(&b);
        }
        
        // 2. Greedy Strategy: Connect Pads to nearest compatible Modules
        for (auto pad : pads) {
            // Get unique astronaut types at this pad
            set<int> neededTypes(pad->astroTypes.begin(), pad->astroTypes.end());
            
            for (int type : neededTypes) {
                Building* bestModule = nullptr;
                double minDistance = 999999.0;
                
                // Find nearest module of this type
                for (auto mod : modules) {
                    if (mod->type == type) {
                        double d = getDistance(pad->x, pad->y, mod->x, mod->y);
                        if (d < minDistance) {
                            minDistance = d;
                            bestModule = mod;
                        }
                    }
                }
                
                if (bestModule) {
                    // Check if route exists
                    Route r = {pad->id, bestModule->id, 0};
                    bool routeExists = existingRoutes.count(r);
                    
                    // COST CALCULATIONS
                    int tubeCost = calculateTubeCost(minDistance);
                    int podCost = 1000;
                    
                    // TRY TO BUILD TUBE
                    if (!routeExists) {
                        if (resources >= tubeCost) {
                            actions.push_back("TUBE " + to_string(pad->id) + " " + to_string(bestModule->id));
                            resources -= tubeCost;
                            
                            // Mark as existing locally so we can add a pod immediately
                            existingRoutes.insert(r);
                            routeExists = true; 
                        }
                    }
                    
                    if (routeExists) {
                        bool hasPod = routesWithPods.count(r);
                        if (!hasPod && resources >= podCost) {
                            // Build a pod that loops: Pad -> Module -> Pad
                            string podCmd = "POD " + to_string(nextPodId++) + " " + 
                                            to_string(pad->id) + " " + 
                                            to_string(bestModule->id) + " " + 
                                            to_string(pad->id);
                            
                            actions.push_back(podCmd);
                            resources -= podCost;
                            
                            // Mark route as having a pod locally
                            routesWithPods.insert(r);
                        }
                    }
                }
            }
        }

        if (actions.empty()) {
            cout << "WAIT" << endl;
        } else {
            for (size_t i = 0; i < actions.size(); i++) {
                cout << actions[i] << (i == actions.size() - 1 ? "" : ";");
            }
            cout << endl;
        }
    }
}

// score 2,656,370