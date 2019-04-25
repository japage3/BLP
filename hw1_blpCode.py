import pandas as pd
import numpy as np
import pyblp
import os
import json
import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore')
def parameters():
	configLoc = os.path.join(os.getcwd(), 'config.json')
	with open(configLoc, 'r') as file:
		config = json.load(file)
		directory = config['directory']
	blpDataFile = 'withWendysBLPData.csv'
	nevoOutput = 'cerealData.csv'
	dataOut = 'withInstrumentsBLP.csv'
	parameters = {'dir': directory, 'file': blpDataFile, 'nevoOutput': nevoOutput, 'dataLoc': dataOut}
	return parameters

def genInstruments(parms):
	fileLoc = os.path.join(parms['dir'], parms['file'])
	data = pd.read_csv(fileLoc)
	formulation = pyblp.Formulation('1 + protein + fat')
	instruments = pyblp.build_blp_instruments(formulation, data)
	# the first set of instruments generated are for other-product-same-manufacturer sumes
	# since we only have solo-product firms, we just drop these to prevent columns of zeroes
	#instruments = instruments[...,5:]
	instNames = ['demand_instruments' + str(x) for x in range(0, instruments.shape[1])]
	instNames[0] = 'demand_instruments0'
	data = pd.concat([data, pd.DataFrame(instruments, index=data.index, columns=instNames)], axis=1)
	return data

def runBLP(parms, data):
	X1_formulation = pyblp.Formulation('0 + prices', absorb='C(product_ids)')
	X2_formulation = pyblp.Formulation('1 + prices + protein + fat')
	product_formulations = (X1_formulation, X2_formulation)
	mc_integration = pyblp.Integration('monte_carlo', size=50, seed=0)
	pr_integration = pyblp.Integration('product', size=5)
	mc_problem = pyblp.Problem(product_formulations, data, integration=mc_integration)
	pr_problem = pyblp.Problem(product_formulations, data, integration=pr_integration)
	# this is not the ideal optimizer and needs to be changed once the code has been proven to work
	opt = pyblp.Optimization('l-bfgs-b')
	#results1 = mc_problem.solve(sigma=np.ones((4, 4)), optimization=bfgs)
	results1 = mc_problem.solve(sigma=np.eye(4), optimization=opt)
	print(results1)
	elasticities = results1.compute_elasticities()
	diversions = results1.compute_diversion_ratios()
	single_market = data['market_ids'] == 18
	plt.colorbar(plt.matshow(elasticities[single_market]))
	plt.colorbar(plt.matshow(diversions[single_market]))
	plt.show()



def fakeBLP():
	product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
	X1_formulation = pyblp.Formulation('0 + prices', absorb='C(product_ids)')
	X2_formulation = pyblp.Formulation('1 + prices + sugar + mushy')
	product_formulations = (X1_formulation, X2_formulation)
	mc_integration = pyblp.Integration('monte_carlo', size=50, seed=0)
	pr_integration = pyblp.Integration('product', size=5)
	mc_problem = pyblp.Problem(product_formulations, product_data, integration=mc_integration)
	pr_problem = pyblp.Problem(product_formulations, product_data, integration=pr_integration)
	bfgs = pyblp.Optimization('bfgs')
	results1 = mc_problem.solve(sigma=np.eye(4), optimization=bfgs)
	elasticities = results1.compute_elasticities()
	diversions = results1.compute_diversion_ratios()
	print(results1)
	print(diversions)
	single_market = product_data['market_ids'] == 'C01Q1'
	plt.colorbar(plt.matshow(diversions[single_market]))




def extractData(parms, data):
	product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
	fileLoc = os.path.join(parms['dir'], parms['nevoOutput'])
	product_data.to_csv(fileLoc)
	fileLoc = os.path.join(parms['dir'], parms['dataLoc'])
	data.to_csv(fileLoc)


def main():
    parms = parameters()
    data = genInstruments(parms)
    runBLP(parms, data)
    #fakeBLP()
    #extractData(parms, data)


if __name__ == '__main__':
    main()