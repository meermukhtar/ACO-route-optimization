from flask import Flask, request, jsonify
import math
import pandas as pd
from sklearn.cluster import KMeans
from typing import List

app = Flask(__name__)

def getShopsForAVendor(shops, vendor):
    vendorShops = []
    for i in range(0, len(shops)):
        contains = False
        supply = {}
        for j in range(0, len(shops[i]['Supply Needed'])):
            if(shops[i]['Supply Needed'][j]['VendorId'] == vendor['id']):
                supply['VendorId']=shops[i]['Supply Needed'][j]['VendorId']
                supply['supplyNeededTrucks']=shops[i]['Supply Needed'][j]['supplyNeededTrucks']
                contains = True
        if(contains == True):
            shopData = {
                'id':shops[i]['id'],
                'Latitude':shops[i]['Latitude'],
                'Longitude':shops[i]['Longitude'],
                'VendorId':supply['VendorId'],
                'supplyNeededTrucks':supply['supplyNeededTrucks'],
                'Name':shops[i]['Name'],
            }
            vendorShops.append(shopData)
    return vendorShops

def calculateNumberOfTrucks(shops):
    sum = 0
    for i in range(0, len(shops)):
        sum = sum + shops[i]['supplyNeededTrucks']
    sum = math.ceil(sum)
    return sum

def getClustersDF(vendorShops):
    df = pd.DataFrame(vendorShops)
    
    df_sorted = df.sort_values(by=['Latitude', 'Longitude'], ascending=[True, True])

    trucksNeeded = calculateNumberOfTrucks(vendorShops)
    
    # Extract latitude and longitude
    X = df_sorted[['Latitude', 'Longitude']]
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=trucksNeeded, random_state=1).fit(X)


    # Add cluster labels to the dataframe
    df_sorted['Cluster'] = kmeans.labels_

    return df_sorted;

def df_to_shop_list(df):
    shop_list = []
    for _, row in df.iterrows():
        shop = Shop(
            id=row['id'],
            name=row['Name'],
            latitude=row['Latitude'],
            longitude=row['Longitude'],
            supply_needed_trucks=row['supplyNeededTrucks'],
            cluster=row['Cluster']
        )
        shop_list.append(shop)
    return shop_list

class GraphNode:
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude
        self.connectedNodes = []
        self.visited = False

    def markVisited(self):
        self.visited = True
        
    def getNearestUnvisitedNode(self)->'GraphNode':
        nearestNode = None
        for i in range(0, len(self.connectedNodes)):
            if(self.connectedNodes[i]["node"].visited == False):
                if(nearestNode == None):
                    nearestNode = self.connectedNodes[i]
                elif(nearestNode["distance"] > self.connectedNodes[i]["distance"]):
                    nearestNode = self.connectedNodes[i]
        if(nearestNode != None):
            nearestNode = nearestNode["node"]
        return nearestNode
                    
    def addNode(self, node:'GraphNode'):
        distance = calculate_distance(self, node);
        self.connectedNodes.append({"node": node, "distance": distance})

    def to_dict(self):
        return {
            "Latitude": self.latitude,
            "Longitude": self.longitude
        }

def calculate_distance(graphNode1:GraphNode, graphNode2:GraphNode):
    # Haversine formula for distance between two lat/lng points
    R = 6371  # Earth radius in km
    lat1, lon1 = math.radians(graphNode1.latitude), math.radians(graphNode1.longitude)
    lat2, lon2 = math.radians(graphNode2.latitude), math.radians(graphNode2.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

class Shop(GraphNode):
    def __init__(self, id, name, latitude, longitude, supply_needed_trucks, cluster):
        GraphNode.__init__(self, latitude, longitude)
        self.id = id
        self.name = name
        self.supply_needed_trucks = supply_needed_trucks
        self.cluster = cluster

    def __repr__(self):
        return f"Shop({self.name}, Cluster {self.cluster})"
    def to_dict(self):
        return {
            "Latitude": self.latitude,
            "Longitude": self.longitude,
            "Truck Supply Needed": self.supply_needed_trucks,
            "Shop Name": self.name,
            "id": self.id
        }

class Vendor(GraphNode):
    def __init__(self, id, name, latitude, longitude, noOfTrucks):
        GraphNode.__init__(self, latitude, longitude)
        self.id = id
        self.name = name
        self.noOfTrucks = noOfTrucks
    def copy(self)->'Vendor':
        return Vendor(self.id, self.name, self.latitude, self.longitude, self.noOfTrucks)
    def __repr__(self):
        return f"Vendor({self.name}, Trucks {self.noOfTrucks})"
    def to_dict(self):
        return {
            "Latitude": self.latitude,
            "Longitude": self.longitude,
            "No of Trucks": self.noOfTrucks,
            "Vendor Name": self.name,
            "id": self.id
        }

class ClusterGraph:
    def __init__ (self, clusterNumber: int, vendor:Vendor, shops:List[Shop]):
        self.vendor = vendor
        self.shops = []
        self.clusterNumber = clusterNumber
        for i in range(0, len(shops)):
            self.addShop(shops[i])

    def addShop(self, shop:Shop):
        self.vendor.addNode(shop)
        shop.addNode(self.vendor)
        for i in range(0, len(self.shops)):
            shop.addNode(self.shops[i])
            self.shops[i].addNode(shop)
        self.shops.append(shop)

    def resetVisit(self):
        self.vendor.visited = False
        for i in range(0, len(self.shops)):
            self.shops[i].visited=False

    def getShortestPathArray(self):
        shortestPath = []
        self.resetVisit()
        currNode = self.vendor
        while(currNode != None):
            shortestPath.append(currNode)
            currNode.markVisited()
            currNode = currNode.getNearestUnvisitedNode()
        return shortestPath
    def __repr__(self):
        return f"Cluster({self.clusterNumber})"
class RoutingHandler:
    def __init__ (self, vendor:Vendor, shops:List[Shop]):
        self.vendor = vendor
        self.clusters = []
        for i in range(0, len(shops)):
            isClusterAvailable = False
            for j in range (0, len(self.clusters)):
                if(self.clusters[j].clusterNumber == shops[i].cluster):
                    self.clusters[j].addShop(shops[i])
                    isClusterAvailable=True
                    break
            if(not isClusterAvailable):
                copyVendor = self.vendor.copy()
                cluster = ClusterGraph(shops[i].cluster, copyVendor, [shops[i]])
                self.clusters.append(cluster)

    def getPaths(self):
        paths = []
        for i in range(0, len(self.clusters)):
            pathGraphNodes = self.clusters[i].getShortestPathArray()
            googleLinkGenerator = GoogleLinkGenerator(pathGraphNodes)
            serializedGraphNodes = [pathGraphNode.to_dict() for pathGraphNode in pathGraphNodes]
            paths.append({"pathLink": googleLinkGenerator.getGoogleLink(), "nodes": serializedGraphNodes})
        return paths

class GoogleLinkGenerator:
    def __init__(self, graphNodes: List[GraphNode]):
        self.graphNodes = graphNodes;

    def getGoogleLink(self):
        urlLink = "https://www.google.com/maps/dir/current+location"
        for i in range(0, len(self.graphNodes)):
            urlLink = urlLink + "/" + str(self.graphNodes[i].latitude) + "," + str(self.graphNodes[i].longitude)
        return urlLink


def getLinksForVendors(vendor, shops):
    vendorShops = getShopsForAVendor(shops, vendor)
    clusterShops = getClustersDF(vendorShops)
    vendor = Vendor(vendor["id"], vendor["Vendor Name"], vendor["Latitude"], vendor["Longitude"], vendor["No of Trucks"])
    shopsList = df_to_shop_list(clusterShops)
    routingGraph = RoutingHandler(vendor, shopsList)
    return routingGraph.getPaths();

# POST endpoint that receives data
@app.route('/api/post-data', methods=['POST'])
def post_data():
    # Get the JSON data sent in the request
    data = request.get_json()
    # Check if 'name' and 'age' exist in the received data
    if 'vendor' in data and 'shops' in data:
        vendor = data['vendor']
        shops = data['shops']

        paths = getLinksForVendors(vendor, shops)


        # Return a success message with the received data
        return jsonify({
            'message': 'Data received successfully!',
            'paths': paths
        }), 200
    else:
        # If 'name' or 'age' is missing, return an error message
        return jsonify({
            'error': 'Invalid data format. Please include both "vendor" and "shops".'
        }), 400

if __name__ == '__main__':
    app.run()
