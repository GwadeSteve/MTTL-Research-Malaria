#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <random>
#include <iomanip>

using namespace std;

// utils
struct Point { double x, y; };

inline double distSq(Point p1, Point p2) {
    return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}

inline int orientation(Point p, Point q, Point r) {
    double val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (abs(val) < 1e-9) return 0;
    return (val > 0) ? 1 : 2;
}

inline bool onSegment(Point p, Point q, Point r) {
    return q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
           q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y);
}

inline bool doIntersect(Point p1, Point q1, Point p2, Point q2) {
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    if (o1 != o2 && o3 != o4) return true;
    if (o1 == 0 && onSegment(p1, p2, q1)) return true;
    if (o2 == 0 && onSegment(p1, q2, q1)) return true;
    if (o3 == 0 && onSegment(p2, p1, q2)) return true;
    if (o4 == 0 && onSegment(p2, q1, q2)) return true;
    return false;
}

inline double ptSegDistSq(Point p, Point a, Point b) {
    double l2 = distSq(a, b);
    if (l2 == 0) return distSq(p, a);
    double t = ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / l2;
    t = max(0.0, min(1.0, t));
    Point proj = { a.x + t * (b.x - a.x), a.y + t * (b.y - a.y) };
    return distSq(p, proj);
}

// const and structures
const int MAX_NODES = 155;
const int MAX_TYPES = 22;

struct Building {
    int id;
    int type;
    Point p;
    int astroCounts[MAX_TYPES]; 
    int totalAstros;
};

struct Route {
    int u, v;
    bool isTele;
};

struct Candidate {
    int u, v;
    int cost;
    bool isTele;
    int geoDist;
    int volume;  // total astronauts this route serves
    double priority; // priority score
};

Building buildings[MAX_NODES];
int buildingCount = 0;
int idToIndex[5000];
int indexToId[MAX_NODES];

vector<Route> existingRoutes;
int R_current;
int nextPodId = 1;
int currentMonth = 0;

// load tracking for balancing
int globalModuleLoad[MAX_NODES];

int adjHead[MAX_NODES];
int adjNext[MAX_NODES * 10];
int adjTo[MAX_NODES * 10];
int adjWeight[MAX_NODES * 10];
int edgeCount;

int dist[MAX_NODES];
int qBuf[MAX_NODES];

void addEdge(int u, int v, int w) {
    adjTo[edgeCount] = v;
    adjWeight[edgeCount] = w;
    adjNext[edgeCount] = adjHead[u];
    adjHead[u] = edgeCount++;
    
    adjTo[edgeCount] = u;
    adjWeight[edgeCount] = w;
    adjNext[edgeCount] = adjHead[v];
    adjHead[v] = edgeCount++;
}

vector<int> targets[MAX_TYPES];
vector<int> activePads;

// fast eval func
double fastEvaluate(const vector<Candidate>& cands, const vector<bool>& active) {
    long long cost = 0;
    
    edgeCount = 0;
    for(int i=0; i<buildingCount; ++i) adjHead[i] = -1;
    
    for(const auto& r : existingRoutes) {
        if (idToIndex[r.u] != -1 && idToIndex[r.v] != -1) {
            addEdge(idToIndex[r.u], idToIndex[r.v], r.isTele ? 0 : 1);
        }
    }
    
    for(size_t i=0; i<cands.size(); ++i) {
        if(active[i]) {
            cost += (cands[i].cost + (cands[i].isTele ? 0 : 1000));
            addEdge(idToIndex[cands[i].u], idToIndex[cands[i].v], cands[i].geoDist);
        }
    }

    if(cost > R_current) return -1e18;

    double score = 0;
    
    // track temp module loads for balancing spots
    int tempModuleLoad[MAX_NODES] = {0};
    
    for(int startNode : activePads) {
        for(int i=0; i<buildingCount; ++i) dist[i] = 999;
        
        int qHead = 0, qTail = 0;
        qBuf[qTail++] = startNode;
        dist[startNode] = 0;
        
        while(qHead < qTail) {
            int u = qBuf[qHead++];
            int d = dist[u];
            if(d >= 20) continue;
            
            for(int e = adjHead[u]; e != -1; e = adjNext[e]) {
                int v = adjTo[e];
                int w = adjWeight[e];
                if(dist[v] > d + w) {
                    dist[v] = d + w;
                    qBuf[qTail++] = v;
                }
            }
        }
        
        const Building& b = buildings[startNode];
        for(int t=1; t<MAX_TYPES; ++t) {
            if(b.astroCounts[t] > 0) {
                int count = b.astroCounts[t];
                
                // search for best target considering distance and load balancing
                int bestTargetIdx = -1;
                double bestTargetScore = -1e18;
                
                for(int targetIdx : targets[t]) {
                    if(dist[targetIdx] < 999) {
                        // compute score = speed pts + balance pts
                        double speedScore = (50.0 - dist[targetIdx]) * count;
                        
                        // balance penalty, weprefer less loaded modules
                        int totalLoad = globalModuleLoad[targetIdx] + tempModuleLoad[targetIdx];
                        double balancePenalty = totalLoad * 0.8; // heavy 
                        
                        double targetScore = speedScore - balancePenalty;
                        
                        if(targetScore > bestTargetScore) {
                            bestTargetScore = targetScore;
                            bestTargetIdx = targetIdx;
                        }
                    }
                }
                
                if(bestTargetIdx != -1) {
                    // compute balance score component
                    int load = globalModuleLoad[bestTargetIdx] + tempModuleLoad[bestTargetIdx];
                    double balanceScore = max(0.0, 50.0 - load) * count;
                    
                    score += (50.0 - dist[bestTargetIdx]) * count + balanceScore;
                    tempModuleLoad[bestTargetIdx] += count;
                }
            }
        }
    }
    
    // resource conservation strategy
    double monthFactor = (currentMonth <= 5) ? 1.5 : //in early month we save more
                        (currentMonth <= 15) ? 0.8 : // in mid periods, we spend aggressively
                        0.3; // in late periods we spen all no need to save
    
    score += (R_current - cost) * 0.03 * monthFactor; // Andre check avec une meilleure heuristique abeg
    
    return score;
}

// candidate generation
vector<Candidate> generateCandidates() {
    vector<Candidate> res;
    
    bool connected[MAX_NODES][MAX_NODES];
    memset(connected, 0, sizeof(connected));
    int degrees[MAX_NODES];
    memset(degrees, 0, sizeof(degrees));
    
    for(const auto& r : existingRoutes) {
        int u = idToIndex[r.u];
        int v = idToIndex[r.v];
        if(u!=-1 && v!=-1) {
            connected[u][v] = connected[v][u] = true;
            if(!r.isTele) { degrees[u]++; degrees[v]++; }
        }
    }
    
    // Calculate volume for each pair (pad+module)
    int pairVolume[MAX_NODES][MAX_NODES];
    memset(pairVolume, 0, sizeof(pairVolume));
    
    for(int i : activePads) {
        const Building& pad = buildings[i];
        for(int t=1; t<MAX_TYPES; ++t) {
            if(pad.astroCounts[t] > 0) {
                for(int j : targets[t]) {
                    pairVolume[i][j] += pad.astroCounts[t];
                }
            }
        }
    }
    
    // tubes ... more selective
    for(int i=0; i<buildingCount; ++i) {
        if(degrees[i] >= 5) continue;
        for(int j=i+1; j<buildingCount; ++j) {
            if(degrees[j] >= 5) continue;
            if(connected[i][j]) continue;
            
            double d2 = distSq(buildings[i].p, buildings[j].p);
            
            // dynamic distance based on volume
            int volume = pairVolume[i][j] + pairVolume[j][i];
            double maxDist2 = (volume > 20) ? 5000.0 : 
                             (volume > 10) ? 3500.0 : 2500.0;
            
            if(d2 > maxDist2) continue;
            
            bool ok = true;
            for(const auto& r : existingRoutes) {
                if(r.isTele) continue;
                if(r.u == buildings[i].id || r.v == buildings[i].id || 
                   r.u == buildings[j].id || r.v == buildings[j].id) continue;
                
                int ru = idToIndex[r.u];
                int rv = idToIndex[r.v];
                if(ru != -1 && rv != -1) {
                    if(doIntersect(buildings[i].p, buildings[j].p, buildings[ru].p, buildings[rv].p)) {
                        ok = false; break;
                    }
                }
            }
            if(!ok) continue;

            for(int k=0; k<buildingCount; ++k) {
                if(k == i || k == j) continue;
                if(ptSegDistSq(buildings[k].p, buildings[i].p, buildings[j].p) < 0.0001) {
                    ok = false; break;
                }
            }
            
            if(ok) {
                int c = (int)(sqrt(d2) * 10);
                double priority = volume * 100.0 - c * 0.5;
                res.push_back({buildings[i].id, buildings[j].id, c, false, 1, volume, priority});
            }
        }
    }
    
    // teleporters ... most aggressive
    for(int i : activePads) {
        const Building& b = buildings[i];
        if(b.totalAstros < 3) continue; // Lower
        
        for(int t=1; t<MAX_TYPES; ++t) {
            if(b.astroCounts[t] > 0) {
                for(int tidx : targets[t]) {
                    int j = tidx;
                    if(connected[i][j]) continue;
                    
                    double d2 = distSq(buildings[i].p, buildings[j].p);
                    int volume = b.astroCounts[t];
                    
                    // More aggressive teleporter criteria
                    bool shouldTeleport = 
                        (d2 > 2500.0 && volume >= 5) ||  // 50km+ with 5+ astronauts
                        (d2 > 1600.0 && volume >= 15) || // 40km+ with 15+ astronauts
                        (volume >= 30);                   // Any 30+ volume route
                    
                    if(shouldTeleport) {
                        double priority = volume * 200.0 - sqrt(d2) * 10.0;
                        res.push_back({buildings[i].id, buildings[j].id, 5000, true, 0, volume, priority});
                    }
                }
            }
        }
    }
    
    // we then sort by priority
    sort(res.begin(), res.end(), [](const Candidate& a, const Candidate& b) {
        return a.priority > b.priority;
    });
    
    return res;
}

// main run
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    srand(42); // we never know :) 

    memset(idToIndex, -1, sizeof(idToIndex));
    memset(globalModuleLoad, 0, sizeof(globalModuleLoad));

    while (true) {
        cin >> R_current;
        
        existingRoutes.clear();
        int numRoutes; cin >> numRoutes;
        for (int i = 0; i < numRoutes; i++) {
            int u, v, c; cin >> u >> v >> c;
            existingRoutes.push_back({u, v, (c==0)});
        }
        
        int numPods; cin >> numPods;
        for (int i = 0; i < numPods; i++) {
            int id, s; cin >> id >> s;
            if(id >= nextPodId) nextPodId = id + 1;
            for(int k=0; k<s; ++k) { int x; cin >> x; }
        }
        
        int numNew; cin >> numNew;
        for(int i=0; i<numNew; ++i) {
            int type; cin >> type;
            int id, x, y;
            int counts[MAX_TYPES] = {0};
            int total = 0;
            
            if(type == 0) {
                int n; cin >> id >> x >> y >> n;
                for(int k=0; k<n; ++k) { int t; cin >> t; counts[t]++; total++; }
            } else {
                cin >> id >> x >> y;
            }
            
            int idx = idToIndex[id];
            if(idx == -1) {
                idx = buildingCount++;
                idToIndex[id] = idx;
                indexToId[idx] = id;
            }
            
            buildings[idx].id = id;
            buildings[idx].type = type;
            buildings[idx].p = {(double)x, (double)y};
            buildings[idx].totalAstros = total;
            memcpy(buildings[idx].astroCounts, counts, sizeof(counts));
        }
        
        currentMonth++;
        
        for(int t=0; t<MAX_TYPES; ++t) targets[t].clear();
        activePads.clear();
        
        for(int i=0; i<buildingCount; ++i) {
            if(buildings[i].type > 0) targets[buildings[i].type].push_back(i);
            else if(buildings[i].totalAstros > 0) activePads.push_back(i);
        }
        
        vector<Candidate> cands = generateCandidates();
        vector<bool> currentSol(cands.size(), false);
        
        // greedy init, we accept top priority candidates
        int budget = R_current;
        for(size_t i=0; i<min(cands.size(), (size_t)20); ++i) {
            if(budget >= cands[i].cost + (cands[i].isTele ? 0 : 1000)) {
                currentSol[i] = true;
                budget -= (cands[i].cost + (cands[i].isTele ? 0 : 1000));
            }
        }
        
        double currentScore = fastEvaluate(cands, currentSol);
        vector<bool> bestSol = currentSol;
        double bestScore = currentScore;
        
        double temp = 300.0; // high init temp
        double alpha = 0.97; // slow
        
        auto startTime = chrono::steady_clock::now();
        
        int maxIter = 3000;
        
        for(int iter=0; iter<maxIter; ++iter) {
            if((iter & 63) == 0) {
                auto now = chrono::steady_clock::now();
                if(chrono::duration_cast<chrono::milliseconds>(now - startTime).count() > 450) break;
            }
            
            vector<bool> nextSol = currentSol;
            
            // neighbor generation
            if(!cands.empty()) {
                int idx = rand() % cands.size();
                nextSol[idx] = !nextSol[idx];
                
                // 0.2 chance of flipping another
                if(rand() % 5 == 0 && cands.size() > 1) {
                    int idx2 = rand() % cands.size();
                    nextSol[idx2] = !nextSol[idx2];
                }
            }
            
            double nextScore = fastEvaluate(cands, nextSol);
            
            if(nextScore > currentScore || exp((nextScore - currentScore)/temp) > ((double)rand()/RAND_MAX)) {
                currentSol = nextSol;
                currentScore = nextScore;
                if(nextScore > bestScore) {
                    bestScore = nextScore;
                    bestSol = nextSol;
                }
            }
            
            temp *= alpha;
        }
        
        // output
        vector<string> cmds;
        vector<Route> finalCheckRoutes = existingRoutes;
        
        for(size_t i=0; i<cands.size(); ++i) {
            if(bestSol[i]) {
                const auto& c = cands[i];
                bool ok = true;
                
                if(!c.isTele) {
                    Point p1 = buildings[idToIndex[c.u]].p;
                    Point p2 = buildings[idToIndex[c.v]].p;
                    
                    int dU=0, dV=0;
                    for(const auto& r : finalCheckRoutes) {
                        if(!r.isTele) {
                            if(r.u == c.u) dU++; if(r.v == c.u) dU++;
                            if(r.u == c.v) dV++; if(r.v == c.v) dV++;
                        }
                    }
                    if(dU >= 5 || dV >= 5) ok = false;
                    
                    if(ok) {
                        for(const auto& r : finalCheckRoutes) {
                            if(r.isTele) continue;
                            if(r.u==c.u || r.u==c.v || r.v==c.u || r.v==c.v) continue;
                            
                            Point rp1 = buildings[idToIndex[r.u]].p;
                            Point rp2 = buildings[idToIndex[r.v]].p;
                            
                            if(doIntersect(p1, p2, rp1, rp2)) { ok = false; break; }
                        }
                    }
                }
                
                int totalCost = c.cost + (c.isTele ? 0 : 1000);
                if(ok && R_current >= totalCost) {
                    if(c.isTele) {
                         cmds.push_back("TELEPORT " + to_string(c.u) + " " + to_string(c.v));
                         R_current -= 5000;
                         finalCheckRoutes.push_back({c.u, c.v, true});
                    } else {
                         cmds.push_back("TUBE " + to_string(c.u) + " " + to_string(c.v));
                         cmds.push_back("POD " + to_string(nextPodId++) + " " + to_string(c.u) + " " + to_string(c.v) + " " + to_string(c.u));
                         R_current -= totalCost;
                         finalCheckRoutes.push_back({c.u, c.v, false});
                    }
                    
                    // upate global load
                    int modIdx = idToIndex[c.v];
                    if(modIdx != -1 && buildings[modIdx].type > 0) {
                        globalModuleLoad[modIdx] += c.volume;
                    }
                }
            }
        }
        
        if(cmds.empty()) cout << "WAIT" << endl;
        else {
            for(size_t i=0; i<cmds.size(); ++i) cout << cmds[i] << (i==cmds.size()-1 ? "" : ";");
            cout << endl;
        }
    }
}