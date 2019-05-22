#import pandas as pd
import numpy as np
import scipy
import csv
import random
import json
import os
import datetime as dt
import pytz

from pyblp.configurations.iteration import Iteration

def setPath():
    configLoc = os.path.join(os.getcwd(), 'config.json')
    with open(configLoc, 'r') as file:
        config = json.load(file)
        path = config['directory']
    return path

def getPricingModel():
    configLoc = os.path.join(os.getcwd(), 'config.json')
    with open(configLoc, 'r') as file:
        config = json.load(file)
        model = config['firmPricingModel']
    return model

def setFileNames():
    names = {'characteristicsCSV': 'BLPTestChars.csv'}
    return names

def genParms():
    np.random.seed(11)
    random.seed(11)
    nMarkets = 300
    pplPerMarket = 500
    marketProbs = {'McDonalds': 0.85, 'BK': 0.75, 'Sweetgreen': 0.55, 'Wendy\'s': 0.55}
    itemDict = {'McDonalds': ["Quarter_Pounder", "McNuggets", 'Southwest_Salad'], 
                'BK': ['Whopper', 'Chicken_Tenders'],
                'Wendy\'s': ['Classic_Double', 'Wendys_Nuggets', 'Garden_Salad'],
                'Sweetgreen': ['Chicken_Bowl', 'Rustic_Salad']
                    }
    reverseItemLookup = genReverseLookup(itemDict)
    itemTypes = {"Quarter_Pounder": 'b', "McNuggets": 'c', 'Southwest_Salad': 's',
                  'Whopper': 'b', 'Chicken_Tenders': 'c',
                   'Classic_Double': 'b', 'Wendys_Nuggets': 'c', 'Garden_Salad': 's',
                    'Chicken_Bowl': 'c', 'Rustic_Salad': 's'}
    firms = ["McDonalds", "BK", "Sweetgreen", "Wendy's"]
    markets = range(0,nMarkets)
    characteristics = {}
    # I tested the ksais and market shares turned out to be extremely sensitive to them. Thus I picked some good looking baseline
    # values and then allowed them to vary market-by-market within a relatively narrow range.
    ksaiDict = {"Quarter_Pounder": 2, "McNuggets": 3.2, 'Southwest_Salad': -2.75,
                  'Whopper': 1.8, 'Chicken_Tenders': 2.75,
                   'Classic_Double': 3, 'Wendys_Nuggets': 2.3, 'Garden_Salad': -8.75,
                    'Chicken_Bowl': 4, 'Rustic_Salad': 0}
    # as with the xi values, so with prices. Shares are very sensitive, so it was best to test them out in advance and keep minimally
    # variable values
    # these prices are superseded by the price generation method (solvePrices) but since I made them somewhat reasonable I keep them
    # as a basis to generate exogenous costs (which we need for Bertrand markups to work when prices *are* generated later)
    priceDict = {"Quarter_Pounder": 2.2, "McNuggets": 1.8, 'Southwest_Salad': 1.6,
                  'Whopper': 3.0, 'Chicken_Tenders': 2,
                   'Classic_Double': 2, 'Wendys_Nuggets': 2, 'Garden_Salad': 1.5,
                    'Chicken_Bowl': 5, 'Rustic_Salad': 7}
    # firm an product codes are used later to output the data and also to order the ownership matrices consistently
    firmCodes = {'McDonalds': 0, 'BK': 1, 'Sweetgreen': 2, 'Wendy\'s': 3}
    prodCodes = {"Quarter_Pounder": 0, "McNuggets": 1, 'Southwest_Salad': 2,
                  'Whopper': 3, 'Chicken_Tenders': 4,
                   'Classic_Double': 5, 'Wendys_Nuggets': 6, 'Garden_Salad': 7,
                    'Chicken_Bowl': 8, 'Rustic_Salad': 9}
    # nests are used for nested logit price derivatives
    nestDict = {"Quarter_Pounder": 0, "McNuggets": 1, 'Southwest_Salad': 2,
                  'Whopper': 0, 'Chicken_Tenders': 1,
                   'Classic_Double': 0, 'Wendys_Nuggets': 1, 'Garden_Salad': 2,
                    'Chicken_Bowl': 1, 'Rustic_Salad': 2}
    # nested logit correlation parameter
    nestLambda = 1
    reverseProdLookup = {v: k for k, v in prodCodes.items()}
    baselineChars = {}
    for firm in firms:
        for item in itemDict[firm]:
            baselineChars[item] = {}
            iType = itemTypes[item]
            if iType == 'b':
                baselineChars[item]['protein'] = np.random.randint(17,21)
                baselineChars[item]['fat'] = np.random.randint(32,37)
            elif iType == 'c':
                baselineChars[item]['protein'] = np.random.randint(16,17)
                baselineChars[item]['fat'] = np.random.randint(25,30)
            elif iType == 's':
                baselineChars[item]['protein'] = np.random.randint(10,15)
                baselineChars[item]['fat'] = np.random.randint(0,4)
    corrMatrix = np.matrix('0.05 0.01; 0.01 0.05 ')
    for firm in firms:
        characteristics[firm] = {}
        for market in markets:
            characteristics[firm][market] = {}
            marketCostShock = np.random.normal(0, 0.5)
            for item in itemDict[firm]:
                characteristics[firm][market][item]= {}
                characteristics[firm][market][item]['protein'] = baselineChars[item]['protein']
                characteristics[firm][market][item]['fat'] = baselineChars[item]['fat']
                #corrShocks = np.random.multivariate_normal([0, 0], corrMatrix)
                #pShock = corrShocks[0]
                #ksShock = corrShocks[1]
                characteristics[firm][market][item]['price'] = priceDict[item] + marketCostShock
                characteristics[firm][market][item]['ksai'] = ksaiDict[item]
                characteristics[firm][market][item]['supply_inst'] = marketCostShock
    betas = [-0.5, 0.8, -0.2] # betas are for price, protein, fat - respectively
    sigmas = [0.2, 0.5, 0.5] 
    nus = [0, 0, 0]
    parameters = {'characteristics': characteristics, 'betas': betas, 'nus': nus, 'sigmas': sigmas, 
                    'nPerMarket': pplPerMarket, 'nMarkets': nMarkets, 'mProbs': marketProbs, 'firmCodes': firmCodes,
                    'prodCodes': prodCodes, 'revLookup': reverseItemLookup, 'itemDict': itemDict, 'revCodes': reverseProdLookup,
                    'nests': nestDict, 'nestingLambda': nestLambda}
    return parameters

def genReverseLookup(itemDict):
    out = {}
    for key in itemDict:
        for entry in itemDict[key]:
            out[entry] = key
    return out

def outChars(parms, path, names):
    fileName = names['characteristicsCSV']
    outFile = os.path.join(path, fileName)
    dta = parms['characteristics']
    with open(outFile, 'w+', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='\"')
        header = ['firm', 'market','product', 'price', 'protein', 'fat']
        writer.writerow(header)
        for f in dta:
            for market in dta[f]:
                for item in dta[f][market]:
                    row = [f]
                    row.append(market)
                    row.append(item)
                    row.append(dta[f][market][item]['price'])
                    row.append(dta[f][market][item]['protein'])
                    row.append(dta[f][market][item]['fat'])
                    writer.writerow(row)

def genPeople(parms, n):
    iden = 0
    markets = range(0, parms['nMarkets'])
    peopleBetas  = {}
    for i in markets:
        peopleBetas[i] = {}
        for j in range(0, n):
            iden += 1
            peopleBetas[i][iden] = {}
            for k, val in enumerate(parms['nus']):
                personalBeta = np.random.normal(parms['nus'][k], parms['sigmas'][k]) + parms['betas'][k]
                peopleBetas[i][iden][k] = personalBeta
    return peopleBetas

def genMarkets(parms):
    # goal is to randomly filter firms from markets so that we get diversity in firm entry into markets
    # 0 firm markets are not allowed - if a market is empty, we pick a random (not Wendy's for obvious reasons) firm to be in it
    num = parms['nMarkets']
    mProbs = parms['mProbs']
    marketFirms = {}
    for i in range(0, num):
        marketFirms[i] = {}
        counter = 0
        for firm in mProbs:
            rand = np.random.uniform(0,1)
            marketFirms[i][firm] = True if rand < mProbs[firm] else False 
            if rand < mProbs[firm]:
                counter += 1 
        if counter == 0:
            rand = np.random.uniform(0,1)
            if rand < 0.33:
                marketFirms[i]['McDonalds'] = True
            elif rand < 0.66:
                marketFirms[i]['BK'] = True
            else:
                marketFirms[i]['Sweetgreen'] = True
    return marketFirms

def simulateChoices(parms, people, markets, dave=True): 
    choices = {}
    for i, mark in enumerate(people):
        choices[mark] = {}          
        for j, person in enumerate(people[i]):
            utilities = []
            products = []
            for resto in parms['characteristics'].keys():
                if (resto != 'Wendy\'s' or dave) and markets[mark][resto]:
                    for prod in parms['characteristics'][resto][mark]:
                        price = parms['characteristics'][resto][mark][prod]['price']
                        protein = parms['characteristics'][resto][mark][prod]['protein']
                        fat = parms['characteristics'][resto][mark][prod]['fat']
                        ksai = parms['characteristics'][resto][mark][prod]['ksai']
                        u = people[mark][person][0]*price + people[mark][person][1]*protein + people[mark][person][2]*fat + ksai + np.random.gumbel()
                        utilities.append(u)
                        products.append(prod)
            # adding the outside option
            utilities.append(0)
            products.append('outside')
            choice = np.argmax(utilities)
            choices[mark][person] = products[choice]
    return choices
    
def marketShares(parms, choices, markets, dave=True):
    shares = {"overall": {}, "market": {}}
    nOverall = 0
    prodSet = set()
    for f in parms['characteristics']:
        for m in parms['characteristics'][f]:
            for p in parms['characteristics'][f][m]:
                prodSet.add((f, p))
    tOverall = {}
    for pair in prodSet:
        # pair[0] is the firm name, pair[1] is the price
        if pair[0] != 'Wendy\'s' or dave:
            tOverall[pair[1]] = 0
    for market in choices:
        nMarket = 0
        tMarket = {}
        for pair in prodSet:
            # pair[0] is the firm name, pair[1] is the price
            if (pair[0] != 'Wendy\'s' or dave) and markets[market][pair[0]]:
                tMarket[pair[1]] = 0
        for iden in choices[market]:
            c = choices[market][iden]
            if c == 'outside':
                nOverall += 1
                nMarket +=1
            else:
                tOverall[c] += 1
                tMarket[c] += 1
                nOverall += 1
                nMarket +=1
        for prod in tMarket:
            tMarket[prod] = float(tMarket[prod])/float(nMarket)
        shares['market'][market] = tMarket
    for prod in tOverall:
        tOverall[prod] = float(tOverall[prod])/float(nOverall)
    shares['overall'] = tOverall
    return shares 

def convertParms(parameters, mark, markNum):
    '''
    unpacks parameters so that they can easily be passed to the objective function
    so this doesn't have to happen with every optimizer iteration
    '''
    miniDict = {} 
    betas = parameters['betas']
    codes = parameters['prodCodes']
    revCodes = parameters['revCodes']
    items = parameters['itemDict']
    prods = []
    prices = []
    xs = []
    for firm in mark:
        if mark[firm] == True:
            for product in items[firm]:
                miniDict[product] = parameters['characteristics'][firm][markNum][product]
                prods.append(product)
    for product in prods:
        x = []
        prices.append(miniDict[product]['price'])
        x.append(miniDict[product]['protein'])
        x.append(miniDict[product]['fat'])
        x.append(miniDict[product]['ksai'])
        xs.append(x)
    sortable = []
    for i, p in enumerate(prods):
        sortable.append((codes[p], xs[i], prices[i]))
    sortable.sort(key = lambda x: x[0])
    positions = []
    outx = []
    outpx = []
    for pair in sortable:
        positions.append(pair[0])
        outx.append(pair[1])
        outpx.append(pair[2])
    return (betas, positions, outx, outpx)

def updateParameters(newPriceDict, parms):
    '''
    accepts newPriceDict which is a market-indexed list of prices and firm identities, goes through the parameters passed and updates
    '''
    revCodes = parms['revCodes']
    lookup = {}
    newParms = parms
    for market in newPriceDict:
        lookup[market] = {}
        pricePos = newPriceDict[market]
        prices = pricePos[0]
        positions = pricePos[1]
        for i, p in enumerate(positions):
            lookup[market][revCodes[p]] = prices[i]
    for firm in parms['characteristics']:
        for market in parms['characteristics'][firm]:
            for product in parms['characteristics'][firm][market]:
                if product in lookup[market]:
                    newParms['characteristics'][firm][market][product]['price'] = lookup[market][product]
                else:
                    newParms['characteristics'][firm][market][product]['price'] = np.nan
    return newParms

def derivatives(alpha, shareI, shareJ, same):
    if same:
        out = -alpha*shareI*(1-shareI)
    else:
        out = alpha*shareI*shareJ
    return out

def nestedDerivatives(alpha, shareI, priceI, shareJ, priceJ, same):
    
    pass

def nestedShares(prices, betas, xs):
    pass

def objFunction(prices, betas, xs, costs, ownership):
    size = len(costs)
    if len(betas) < 4:
        betas.append(1)
    alpha = betas[0]
    betas = np.array(betas)
    oldPrices = np.array(prices)
    prices = np.array(prices)[np.newaxis]
    costs = np.array(costs)
    xs = np.matrix(xs)
    bigX = np.concatenate((prices.T, xs), axis=1)
    pre = bigX @ betas
    expVec = np.exp(pre)
    summ = np.sum(expVec) + 1
    shares = expVec/summ
    matrix = np.zeros((size, size))
    prices = prices[0]
    i = 0
    shares = np.asarray(shares)[0]
    for i in range(0, size):
        for j in range(0, size):
            if ownership[i][j] == 1:
                if i == j:
                    matrix[i][j] = derivatives(alpha, shares[i], shares[j], True) 
                else:
                    matrix[i][j] = derivatives(alpha, shares[i], shares[j], False)
    objVec = shares - matrix @ (prices - costs)
    objective = np.linalg.norm(objVec, 1)
    return objective
    
def logitPrices(parms, markets, nested=False):
    '''
    Solving for prices as a (nested) logit using scipy optimizer
    '''
    marketPrices = {}
    for mark in markets:
        if mark % 50 == 0:
            print('computed market {}...'.format(mark))
        ownMat = genOwnMat(parms, markets[mark])
        costs = genCosts(parms, markets[mark], mark)
        miniParms = convertParms(parms, markets[mark], mark)
        betas = miniParms[0]
        positions = miniParms[1]
        xs = miniParms[2]
        priceGuess = miniParms[3]
        bds = scipy.optimize.Bounds(costs, np.inf)
        result = scipy.optimize.minimize(objFunction, priceGuess, args=(betas, xs, costs, ownMat), bounds=bds)
        newPrices = result.get('x')
        marketPrices[mark] = [newPrices, positions]
    parms = updateParameters(marketPrices, parms)
    return parms

def solvePrices(parms, markets):
    '''
     pyblp has a built in method for solving for price equilibria according to Morrow and Skerlos' (2011) method
     this is slightly different from our method of computing shares 
    
    '''
    out = parms
    iteration = Iteration('simple', {'atol': 1e-12}) 
    for mark in markets:
        ownMat = genOwnMat(parms, markets[mark])
        costs = genCosts(parms, markets[mark], mark)

    assert False, "solvePrices method not complete"
    return out

def genOwnMat(parms, mark):
    firmCodes = parms['firmCodes']
    prodCodes = parms['prodCodes']
    invProdCodes = {v: k for k, v in prodCodes.items()}
    reverseLookup = parms['revLookup']
    itemDict = parms['itemDict']
    row = []
    for firm in parms['characteristics']:
        if mark[firm] == True:
            for item in itemDict[firm]:
                row.append(prodCodes[item])
    row.sort()
    size = len(row)
    outMat = np.zeros((size, size))
    for i, num1 in enumerate(row):
        for j, num2 in enumerate(row):
            if reverseLookup[invProdCodes[i]] == reverseLookup[invProdCodes[j]]: 
                outMat[i, j] = 1
    return outMat

def genCosts(parms, mark, markNum):
    preOut = []
    prodCodes = parms['prodCodes']
    itemDict = parms['itemDict']
    for firm in parms['characteristics']:
        if mark[firm] == True:
            for item in itemDict[firm]:
                px = parms['characteristics'][firm][markNum][item]['price']
                cost = px * .75
                preOut.append((prodCodes[item], cost))
    preOut.sort(key = lambda x: x[0])
    almostOut = [cost for i, cost in preOut]
    out = np.array(almostOut)
    return out

def generateEstimationData(shares, parms, path, fileName):
    outFile = os.path.join(path, fileName)
    firmCodes = parms['firmCodes']
    prodCodes = parms['prodCodes']
    reverseLookup = parms['revLookup']
    with open(outFile, 'w+', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='\"')
        header = ['market_ids','firm_ids', 'product_ids', 'shares', 'prices', 'protein', 'fat']
        writer.writerow(header)
        for market in shares['market']:
            for prod in shares['market'][market]:
                row = [market]
                firm = reverseLookup[prod]
                row.append(firmCodes[firm])
                row.append(prodCodes[prod])
                row.append(shares['market'][market][prod])
                row.append(parms['characteristics'][firm][market][prod]['price'])
                row.append(parms['characteristics'][firm][market][prod]['protein'])
                row.append(parms['characteristics'][firm][market][prod]['fat'])
                writer.writerow(row)

def genFullDataForEstimation():
    start = dt.datetime.now(pytz.utc)
    path = setPath()
    fns = setFileNames()
    parms = genParms()
    outChars(parms, path, fns)
    # need to gen prices endogenously *after* the markets are simulated 
    people = genPeople(parms, parms['nPerMarket'])
    # randomizes which firms are present in each market
    markets = genMarkets(parms)
    pricing = getPricingModel()
    if pricing == "full":
        parms = solvePrices(parms, markets)
    elif pricing == "Logit":
        parms = logitPrices(parms, markets)
    # 'dave' is the Wendy's parameter, to include wendy's products in the choice set or not
    dave = True
    newChoices = simulateChoices(parms, people, markets, dave=dave)
    newShares = marketShares(parms, newChoices, markets, dave=dave)
    print(newShares['overall'])
    generateEstimationData(newShares, parms, path, 'withWendysBLPData.csv')

def genWithWithoutWendys():
    start = dt.datetime.now(pytz.utc)
    path = setPath()
    fns = setFileNames()
    parms = genParms()
    outChars(parms, path, fns)
    # need to gen prices endogenously *after* the markets are simulated 
    people = genPeople(parms, parms['nPerMarket'])
    # randomizes which firms are present in each market
    markets = genMarkets(parms)
    parms = solvePrices(parms, markets)
    # 'dave' is the Wendy's parameter, to include wendy's products in the choice set or not
    dave = False
    choices = simulateChoices(parms, people, markets, dave=dave)
    shares = marketShares(parms, choices, markets, dave=dave)
    print(shares['overall'])
    dave = True
    newChoices = simulateChoices(parms, people, markets, dave=dave)
    newShares = marketShares(parms, newChoices, markets, dave=dave)
    sharesAsCSV(shares, path, 'noWendys.csv')
    sharesAsCSV(newShares, path, 'withWendys.csv')
    print('generated {} markets each with {} observations in {} minutes'.format(parms['nMarkets'], parms['nPerMarket'], \
    (dt.datetime.now(pytz.utc) - start).seconds/60))


def main():
    genFullDataForEstimation()

if __name__ == '__main__':
    main()