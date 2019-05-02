#import pandas as pd
import numpy as np
import csv
import random
import os
import datetime as dt
import pytz

def setPath():
    configLoc = os.path.join(os.getcwd(), 'config.json')
    with open(configLoc, 'r') as file:
        config = json.load(file)
        path = config['directory']
    return path

def genParms():
    np.random.seed(10)
    random.seed(10)
    nMarkets = 10000
    items = ["McDonalds", "BK", "Sweetgreen"]
    markets = range(0,nMarkets)
    characteristics = {}
    ksais = {}
    ksais['outside'] = {}
    for item in items:
        characteristics[item] = {}
        ksais[item] = {}
        for market in markets:
            characteristics[item][market] = {}
            if item == "Sweetgreen":
                characteristics[item][market]['price'] = np.random.randint(4,6)
                characteristics[item][market]['protein'] = np.random.randint(15,20)
                characteristics[item][market]['salt'] = np.random.uniform(5,10)
                characteristics[item][market]['fat'] = 0
                ksais[item][market] = np.random.normal(3,1)
            else:
                characteristics[item][market]['price'] = np.random.randint(2,4)
                characteristics[item][market]['protein'] = np.random.randint(15,25)
                characteristics[item][market]['salt'] = np.random.uniform(30,50)
                characteristics[item][market]['fat'] = np.random.uniform(5,10)
                ksais[item][market] = np.random.normal(3,1)
            ksais['outside'][market] = 0
    betas = [1.0, -0.45, -0.15]
    sigmas = [1.5,0.6,0.6]
    nus = []
    for i in range(0,3):
        nus.append(np.random.normal(0,.2))

    parameters = {'characteristics': characteristics, 'ksais':ksais, 'betas': betas, 'price': -2.5, 'nus': nus, 'sigmas': sigmas, 'nPerMarket': 1000, 'numMarkets':nMarkets}
    return parameters
    
def addWendys(parms):
    chars = parms['characteristics']
    ksais = parms['ksais']
    chars['Wendys'] = {}
    ksais['Wendys'] = {}
    for market in chars['McDonalds']:
        chars['Wendys'][market] = {}
        chars['Wendys'][market]['price'] = np.random.randint(2,4)
        chars['Wendys'][market]['protein'] = np.random.randint(15,25)
        chars['Wendys'][market]['salt'] = np.random.uniform(30,50)
        chars['Wendys'][market]['fat'] = np.random.uniform(5,10)
        ksais['Wendys'][market] = np.random.normal(3,1)
    parameters = {'characteristics': chars, 'ksais':ksais, 'betas': parms['betas'], 'price': parms['price'], 'nus': parms['nus'], 'sigmas': parms['sigmas']}
    return parameters

def genPeople(parms, n):
    iden = 0
    markets = range(0, parms['numMarkets'])
    peopleBetas  = {}
    for i in markets:
        peopleBetas[i] = []
        for j in range(0, n):
            iden += 1
            indBetas = [iden, parms['price']]
            for k, val in enumerate(parms['nus']):
                indBetas.append(np.random.normal(parms['nus'][k], parms['sigmas'][k]) + parms['betas'][k])
            peopleBetas[i].append(indBetas)
    return peopleBetas

def simulateChoices(parms, people): 
    choices = {}
    for i, mark in enumerate(people):
        choices[mark] = {}          
        for j, person in enumerate(people[i]):
            utilities = []
            for resto in parms['characteristics'].keys():
                price = parms['characteristics'][resto][mark]['price']
                protein = parms['characteristics'][resto][mark]['protein']
                salt = parms['characteristics'][resto][mark]['salt']
                fat = parms['characteristics'][resto][mark]['fat']
                ksai = parms['ksais'][resto][mark] 
                u = parms['price']*price + people[mark][j][2]*protein + people[mark][j][3]*salt + people[mark][j][4]*fat + ksai + np.random.gumbel()
              #  print([price, people[mark][j][2], protein, people[mark][j][3], salt, people[mark][j][4], fat, ksai, 10*np.random.gumbel()])
                utilities.append(u)
            outside = parms['ksais']['outside'][mark]
            utilities.append(outside)
            #if j > 2:
             #  assert False
            choice = np.argmax(utilities)
            choices[mark][people[mark][j][0]] = choice 
    return choices
    
def marketShares(choices, wendys=False):
    shares = {"overall": {}, "market": {}}
    nOverall = 0
    if wendys:
        tOverall = {'McDonalds': 0, 'BK': 0, 'Sweetgreen': 0, 'Wendys': 0}
    else:
        tOverall = {'McDonalds': 0, 'BK': 0, 'Sweetgreen': 0}
    for market in choices:
        nMarket = 0
        if wendys:
            tMarket = {'McDonalds': 0, 'BK': 0, 'Sweetgreen': 0, 'Wendys': 0}
        else:
            tMarket = {'McDonalds': 0, 'BK': 0, 'Sweetgreen': 0}
        for iden in choices[market]:
            c = choices[market][iden]
            if c == 0:
                tOverall['McDonalds'] += 1
                tMarket['McDonalds'] += 1
            elif c == 1:
                tOverall['BK'] += 1
                tMarket['BK'] += 1
            elif c == 2:
                tOverall['Sweetgreen'] += 1
                tMarket['Sweetgreen'] += 1
            elif c == 3 and wendys:
                tOverall['Wendys'] += 1
                tMarket['Wendys'] += 1
            nOverall += 1
            nMarket +=1
        for store in tMarket:
            tMarket[store] = float(tMarket[store])/float(nMarket)
        shares['market'][market] = tMarket
    for store in tOverall:
        tOverall[store] = float(tOverall[store])/float(nOverall)
    shares['overall'] = tOverall
    return shares  

def sharesAsCSV(shares, path, fileName):
    outFile = os.path.join(path, fileName)
    with open(outFile, 'w+', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='\"')
        if len(shares['overall'].keys()) == 3:
            header = ['Market', 'McDonalds', 'BK', 'Sweetgreen']
            firstRow = ['Total']
            for place in shares['overall']:
                firstRow.append('{0:.3g}'.format(shares['overall'][place]))
        else:
            header = ['Market', 'McDonalds', 'BK', 'Sweetgreen', 'Wendy\'s']
            firstRow = ['Total']
            for place in shares['overall']:
                firstRow.append('{0:.3g}'.format(shares['overall'][place]))
        writer.writerow(header)
        writer.writerow(firstRow)
        for item in shares['market']:
            row = [item]
            for place in shares['market'][item]:
                row.append('{0:.3g}'.format(shares['market'][item][place]))
            writer.writerow(row)

def choicesAsCSV(choices, parms, path, fileName):
    cd = {0: 'McDonalds', 1:'BK', 2:'Sweetgreen', 3:'Wendy\'s', 4: 'outside'}
    outFile = os.path.join(path, fileName)
    with open(outFile, 'w+', newline='') as file: 
        writer = csv.writer(file, delimiter=',', quotechar='\"')
        if len(parms.keys()) == 3:
            header = ['ID', 'Market', 'choice', 'McDonalds_Price', 'McDonalds_Protein', 'McDonalds_Fat', 'McDonalds_Salt', 'BK_Price',\
             'BK_Protein', 'BK_Fat', 'BK_Salt', 'Sweetgreen_Price', 'Sweetgreen_Protein', 'Sweetgreen_Fat', 'Sweetgreen_Salt']
        else:
            header = ['ID', 'Market', 'choice', 'McDonalds_Price', 'McDonalds_Protein', 'McDonalds_Fat', 'McDonalds_Salt', 'BK_Price',\
             'BK_Protein', 'BK_Fat', 'BK_Salt', 'Sweetgreen_Price', 'Sweetgreen_Protein', 'Sweetgreen_Fat', \
             'Sweetgreen_Salt',  'Wendys_Price', 'Wendys_Protein', 'Wendys_Fat', 'Wendys_Salt' ]
        writer.writerow(header)
        for market in choices:
            for iden in choices[market]:
                row = [iden, market, cd[choices[market][iden]], parms['characteristics']['McDonalds'][market]['price'], parms['characteristics']['McDonalds'][market]['protein'], \
                 parms['characteristics']['McDonalds'][market]['salt'], parms['characteristics']['McDonalds'][market]['fat'], parms['characteristics']['BK'][market]['price'], \
                 parms['characteristics']['BK'][market]['protein'], parms['characteristics']['BK'][market]['salt'], parms['characteristics']['BK'][market]['fat'], \
                 parms['characteristics']['Sweetgreen'][market]['price'], parms['characteristics']['Sweetgreen'][market]['protein'], parms['characteristics']['Sweetgreen'][market]['salt'],\
                 parms['characteristics']['Sweetgreen'][market]['fat']]
                if len(parms.keys()) != 3:
                    rowAppend = [parms['characteristics']['Wendys'][market]['price'], parms['characteristics']['Wendys'][market]['protein'], parms['characteristics']['Wendys'][market]['salt'], \
                     parms['characteristics']['Wendys'][market]['fat']]
                    for thing in rowAppend:
                        row.append(thing)
                writer.writerow(row)

def generateEstimationData(shares, parms, path, fileName):
    outFile = os.path.join(path, fileName)
    revFirmLookup = {'McDonalds': 0, 'BK': 1, 'Sweetgreen': 2, 'Wendys': 3}
    pLookup = {'McDonalds': 0, 'BK': 1, 'Sweetgreen': 2, 'Wendys': 3}
    with open(outFile, 'w+', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='\"')
        header = ['market_ids','firm_ids', 'product_ids', 'shares', 'prices', 'protein', 'fat', 'salt']
        writer.writerow(header)
        for market in shares['market']:
            for prod in shares['market'][market]:
                row = [market]
                row.append(revFirmLookup[prod])
                row.append(pLookup[prod])
                row.append(shares['market'][market][prod])
                row.append(parms['characteristics'][prod][market]['price'])
                row.append(parms['characteristics'][prod][market]['protein'])
                row.append(parms['characteristics'][prod][market]['fat'])
                row.append(parms['characteristics'][prod][market]['salt'])
                writer.writerow(row)

def main():
    start = dt.datetime.now(pytz.utc)
    parms = genParms()
    people = genPeople(parms, parms['nPerMarket'])
    choices = simulateChoices(parms, people)
    shares = marketShares(choices)
    print(shares['overall'])
    newParms = addWendys(parms)
    newChoices = simulateChoices(newParms, people)
    newShares = marketShares(newChoices, wendys=True)
    print(newShares['overall'])
    # print(newShares['overall'])
    path = setPath()
    sharesAsCSV(shares, path, 'noWendys.csv')
    sharesAsCSV(newShares, path, 'withWendys.csv')
    #choicesAsCSV(choices, parms, path, 'noWendysChoices.csv')
    #choicesAsCSV(newChoices, newParms, path, 'withWendysChoices.csv')
    generateEstimationData(newShares, newParms, path, 'withWendysBLPData.csv')
    print('generated {} markets each with {} observations in {} minutes'.format(parms['numMarkets'], parms['nPerMarket'], \
        (dt.datetime.now(pytz.utc) - start).seconds/60))



if __name__ == '__main__':
    main()