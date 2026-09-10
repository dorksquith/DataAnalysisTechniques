# VL or VR?
import uproot
import numpy as np
import pandas as pd
import awkward as ak
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


mpl.rcParams['legend.fontsize']= 18
mpl.rcParams['axes.labelsize']= 16
mpl.rcParams['axes.titlesize']= 16
mpl.rcParams['xtick.labelsize']=16
mpl.rcParams['ytick.labelsize']=16

# Choose a salary point for plots
# Note that the distributions will not change based on salary, but the payment amounts will.
test_salary=50

# CONSTANTS
c_maxyears  = 20    # Only the most recent 20 years count
c_maxCR     = 22.53 # Maximum SRP is £22,530
c_minyears  = 2     # Minimum years of service for SRP
c_maxweekly = 0.751 # Maximum weekly pay considered for SRP is £751
c_maxVR     = 95    # Maximum VR is £95,000
c_multVR    = 1.5   # Multiplier for VR
c_multVL_a  = 2.1   # option a) 2.1 weeks pay times years of service
c_multVL_b  = 15.6  # option b) 3.6 months pay = 15.6 weeks pay

def multiplier(agex):
	m = 0.5 if agex < 22 else 1 if agex < 41 else 1.5
	return m

def monthly_to_weekly(monthly):
	return monthly * 12 / 52

# return 1 or identity matrix depending on type of arg
def identity(arg):
	scalarmode = isinstance(arg, float) or isinstance(arg, int)

	I = 1 if scalarmode else np.ones(shape=arg.shape)

	return I


#===========================================================
# Gross salary = Basic - Pension
# USS basic member contribution rate is 6.1% of basic
#===========================================================
def gross_from_basic(basic_annual, pension=6.1):
	pension_deducted = pension/100 * basic_annual
	gross_annual = basic_annual - pension_deducted
	return round(gross_annual, 2)

#===========================================================
# Net salary from Basic salary stolen and adapted from internet
#===========================================================
def net_from_basic(basic_annual, pension=6.1):

	pension_deducted = pension/100 * basic_annual

	gross_annual = basic_annual - pension_deducted

	personal_allowance = 12570
	
	tax_brackets = [ (50270, 0.20), (float('inf'), 0.40) ]
	ni_brackets  = [ (50270, 0.08), (float('inf'), 0.02) ]

	taxable = max(0.0, basic_annual - personal_allowance)
	
	total_tax = 0.0
	previous_limit = 0.0

	for limit, rate in tax_brackets:
		if taxable > previous_limit:
			taxable_in_bracket = min(taxable, limit) - previous_limit
			total_tax += taxable_in_bracket * rate
			previous_limit = limit
		else:
			break

	total_ni=0
	previous_limit = 0.0
	for limit, rate in ni_brackets:
		if taxable > previous_limit:
			taxable_in_bracket = min(taxable, limit) - previous_limit
			total_ni += taxable_in_bracket * rate
			previous_limit = limit
		else:
			break

	return round(gross_annual  - total_tax - total_ni, 2), total_tax, total_ni


def run_tax_check():
	basic_spinal_salary = 69488
	# monthly basic 5,790.67
	# monthly pension 353.23
	# monthly gross 5,437.44
	# monthly tax 1,127.27
	# monthly nic 276.25
	# monthly net 4,033.92

	# if no pension, put 0
	pension_contrib=6.1 # USS basic pension contribution rate

	print( f"\nmonthly basic: {(basic_spinal_salary/12):.2f}"  )
	print( f"\nmonthly pension: {((pension_contrib/100 * basic_spinal_salary)/12):.2f}"  )

	gross = gross_from_basic(basic_spinal_salary, pension=pension_contrib)
	print( f"\nmonthly gross: {(gross/12):.2f}"  )

	net,tax,ni = net_from_basic(basic_spinal_salary, pension=pension_contrib)

	print( f"\nmonthly tax: {(tax/12):.2f}")
	print( f"\nmonthly ni: {(ni/12):.2f}")
	
	print( f"\nmonthly net = {(net/12):.2f}"  )

'''
monthly basic: 5790.67  <= yes
monthly pension: 353.23 <= yes
monthly gross: 5437.44  <= yes
monthly tax: 1059.43    NO -> should be 1,127.27, so is too low by £67.84
monthly ni: 346.21      NO -> should be 276.25, so is too high  by £69.96
monthly net = 4031.79   almost right.

Conclusion: something to do with tax year switch, previous SL deductions, or something wrong with code.
'''


#===========================================================
# Entitled weeks is a function of age and years of service
#===========================================================

def entitled_weeks(age, nyears):
	w = 0
	for y in range(0, nyears ):
		if y >= c_maxyears:
			break
		a = age - y
		m = multiplier(a) # 0.5, 1, 1.5 depending on age in year of service
		w += m
	return w

#===========================================================
# 2.5 months = 11 weeks additional pay for VR versus VL
# based on leaving date for VR being later than for VL
#===========================================================
def extra_pay_from_vr(monthly):
	basic = 2.5 * monthly
	net = net_from_basic(basic)[0]
	return net

#===========================================================
# Compulsory Reduncancy (Statutory Redundancy Payment)
# functions can take scalar or vector arguments
#===========================================================
# weekly: basic spinal weekly salary in £k
# nweeks: entitled_weeks
# nyears: years of service
def CR(weekly, nweeks, nyears):

	I = identity(weekly)

	# cap the weekly pay at c_maxweekly
	weekly = np.minimum( weekly, c_maxweekly * I)

	# zero payment if min years requirement not met
	if np.isscalar(nyears):
		if(nyears < c_minyears):
			return 0
	else:
		weekly[ nyears < c_minyears ] = 0

	return np.minimum( nweeks * weekly, c_maxCR * I)

#===========================================================
# Voluntary redundancy payment is 1.5 * weekly_pay * entitled_weeks capped at 95k
#===========================================================
def VR( weekly, nweeks):

	I = identity(weekly)

	return np.minimum( c_multVR * nweeks * weekly, c_maxVR * I )


#===========================================================
# Voluntary Leavers payment is the larger of 
# a) 2.1 weeks pay * years of service
# b) 3.6 months pay = 15.6 weeks pay
# Note that this is independent of age.
# capped at 95k
#===========================================================
def VL(weekly, nyears):

	I = identity(weekly)

	VL_a = c_multVL_a * weekly * nyears
	VL_b = c_multVL_b * weekly

	return np.minimum( np.maximum( VL_a, VL_b), c_maxVR * I)

def VLa(weekly, nyears):

	I = identity(weekly)

	VL_a = c_multVL_a * weekly * nyears

	return np.minimum( VL_a, c_maxVR * I)

def VLb(weekly, nyears):

	I = identity(weekly)

	VL_b = c_multVL_b * weekly 

	return np.minimum( VL_b, c_maxVR * I)

#===========================================================
# Calculate the CR, VR, and VL26 payments
#===========================================================
def calculate(nyears, age, monthly):

	weekly = monthly_to_weekly(monthly)

	nweeks = entitled_weeks(age,nyears)
	
	calc_cr = CR(weekly, nweeks, nyears)
	calc_vr = VR(weekly, nweeks)
	calc_vl = VL(weekly, nyears)
	
	

	return calc_cr, calc_vr, calc_vl

def print_examples():
	#===========================================================
	# Print some example scenarios
	#===========================================================
	print(f"\nFirst an example for a made-up young person.")

	# Example 0
	input_years = 1
	input_age = 22
	input_basic_monthly = 25/12
	print(f"\nExample 0:\nAlice is 22 and has worked at Sussex for 1 year on a current salary of £25k")
	calc_cr, calc_vr, calc_vl= calculate(input_years, input_age, input_basic_monthly)

	calc_vr_extra = calc_vr + extra_pay_from_vr( input_basic_monthly )
	print(f"calc_vr_extra: {calc_vr_extra}")


	print(f"CR=£{ calc_cr:.2f}k, VR=£{ calc_vr:.2f}k, VR+=£{calc_vr_extra:.2f}k, VL=£{ calc_vl:.2f}k")
	print(f"Alice is very likely to apply for VL, because there is no CR and the VR would barely cover a month's rent.")

	print(f"For Alice, VLa = {VLa(monthly_to_weekly(input_basic_monthly),input_years):.2f} and VLb = {VLb(monthly_to_weekly(input_basic_monthly),input_years):.2f}")

	print(f"\nThese three examples are from the SharePoint CC site. Note that all three have better outcomes for VL than VR.")

	# Example 1
	input_years = 7
	input_age = 35
	input_basic_monthly = 40/12
	print(f"\nExample 1: 7 years of service, age 35, salary £40k")
	calc_cr, calc_vr, calc_vl= calculate(input_years, input_age, input_basic_monthly)
	calc_vr_extra = calc_vr + extra_pay_from_vr( input_basic_monthly )

	print(f"CR=£{ calc_cr:.2f}k, VR=£{ calc_vr:.2f}k, VR+=£{calc_vr_extra:.2f}k, VL=£{ calc_vl:.2f}k")

	print(f"For 1, VLa = {VLa(monthly_to_weekly(input_basic_monthly),input_years):.2f} and VLb = {VLb(monthly_to_weekly(input_basic_monthly),input_years):.2f}")


	# Example 2
	input_years = 18
	input_age = 50
	input_basic_monthly = 50/12
	print(f"\nExample 2: 18 years of service, age 50, salary £50k")
	calc_cr, calc_vr, calc_vl=calculate(input_years, input_age, input_basic_monthly)
	calc_vr_extra = calc_vr + extra_pay_from_vr( input_basic_monthly )
	print(f"CR=£{ calc_cr:.2f}k, VR=£{ calc_vr:.2f}k, VR+=£{calc_vr_extra:.2f}k, VL=£{ calc_vl:.2f}k")

	print(f"For 2, VLa = {VLa(monthly_to_weekly(input_basic_monthly),input_years):.2f} and VLb = {VLb(monthly_to_weekly(input_basic_monthly),input_years):.2f}")

	# Example 3
	input_years = 35
	input_age = 65
	input_basic_monthly = 60/12
	print(f"\nExample 3: 35 years of service, age 65, salary £60k")
	calc_cr, calc_vr, calc_vl=calculate(input_years, input_age, input_basic_monthly)
	calc_vr_extra = calc_vr + extra_pay_from_vr( input_basic_monthly )
	print(f"CR=£{ calc_cr:.2f}k, VR=£{ calc_vr:.2f}k, VR+=£{calc_vr_extra:.2f}k, VL=£{ calc_vl:.2f}k")

	print(f"For 3, VLa = {VLa(monthly_to_weekly(input_basic_monthly),input_years):.2f} and VLb = {VLb(monthly_to_weekly(input_basic_monthly),input_years):.2f}")


	print(f"\nAnd some examples for made up people for whom VL is a worse deal:")


	# Example 4
	input_years = 10
	input_age = 50
	input_basic_monthly = 70/12
	print(f"\nExample 4:\nBob is 50 and has worked at Sussex for 10 years on a current salary of £70k")

	calc_cr, calc_vr, calc_vl=calculate(input_years, input_age, input_basic_monthly)
	calc_vr_extra = calc_vr + extra_pay_from_vr( input_basic_monthly )
	print(f"CR=£{ calc_cr:.2f}k, VR=£{ calc_vr:.2f}k, VR+=£{calc_vr_extra:.2f}k, VL=£{ calc_vl:.2f}k")

	print(f"Bob will not be considering applying for VL, because it is LESS than VR.")

	print(f"For Bob, VLa = {VLa(monthly_to_weekly(input_basic_monthly),input_years):.2f} and VLb = {VLb(monthly_to_weekly(input_basic_monthly),input_years):.2f}")

	# Example 5
	input_years = 18
	input_age = 60
	input_basic_monthly = 45/12
	print(f"\nExample 5:\nCristobel is 60 and has worked at Sussex for 18 years on a current salary of £45k")

	calc_cr, calc_vr, calc_vl=calculate(input_years, input_age, input_basic_monthly)
	calc_vr_extra = calc_vr + extra_pay_from_vr( input_basic_monthly )
	print(f"CR=£{ calc_cr:.2f}k, VR=£{ calc_vr:.2f}k, VR+=£{calc_vr_extra:.2f}k, VL=£{ calc_vl:.2f}k")

	print(f"Cristobel will not be considering applying for VL, because it is LESS than VR.\n")

	print(f"For Cristobel, VLa = {VLa(monthly_to_weekly(input_basic_monthly),input_years):.2f} and VLb = {VLb(monthly_to_weekly(input_basic_monthly),input_years):.2f}")


#====================================================
# 
# PLOTTING PART
#
#====================================================
def make_plots():
	salaries=[25,30,40,50,60,70,80,90,100]

	for test_salary in salaries:

		# 1D array: ages
		ages = np.arange(18,76,1)

		# 1D array: years of service
		yoss = np.arange(0,40,1)

		# calculate weekly pay
		weekly = test_salary / 52
		monthly = test_salary / 12

		#====================================================
		# 2D arrays:
		# W: entitled weeks in 2D array shape
		# P: weekly pay in 2D array shape
		# impmask: impossible values (yos cannot exceed age-18) filled with 0, possible values filled with 1
		#====================================================
		W = []
		P = []
		impmask = []

		for age in ages:
			wrow=[]
			prow=[]
			improw=[]
			for yos in yoss:
				if age-yos < 18:
					wy = 1
					py = 1
					iy = 0
				else:
					wy = entitled_weeks(age, yos)
					py = weekly
					iy = 1

				wrow.append(wy)
				prow.append(py)
				improw.append(iy)

			W.append(wrow)
			P.append(prow)
			impmask.append(improw)

		# convert 2D lists to numpy arrays

		W=np.array(W)
		P=np.array(P)
		impmask = np.array(impmask)

		#====================================================
		# 2D arrays of X = ages, Y = years of service
		#====================================================
		# 2D arrays (NxN squares, each row a copy of the 1D arrays we defined above)
		X,Y = np.meshgrid(ages, yoss)


		#====================================================
		# Calculate payments for the three different scenarios
		#====================================================

		# Y.T : Pass the Transpose ofthe "years of service" array (rows <=> columns)

		# Compulsory (statutory redundancy pay) 
		calc_cr = CR(P, W, Y.T)
		# in terms of month of salary
		calc_cr_sal = calc_cr / (test_salary/12 * np.ones(shape=calc_cr.shape) )

		# Voluntary redundancy payment
		calc_vr = VR(P, W)
		# in terms of month of salary
		calc_vr_sal = calc_vr / (test_salary/12 * np.ones(shape=calc_vr.shape) )

		# Voluntary redundancy payment plus 2.5 months pay (different timings for VR and VL)
		calc_vr_extra = calc_vr + extra_pay_from_vr( monthly )
		calc_vr_extra_sal = calc_vr_extra / (test_salary/12 * np.ones(shape=calc_vr_extra.shape) )

		# Voluntary Leavers 26 payment
		calc_vl = VL(P, Y.T)
		# in terms of month of salary
		calc_vl_sal = calc_vl / (test_salary/12 * np.ones(shape=calc_vl.shape) )

		#====================================================
		# Make plots
		#====================================================

		# set up figure: 4 sublots
		fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(18, 12), layout='tight',sharex=False, sharey=False)

		# figure name for this test_salary
		#figname = f"VLminusVR_{test_salary}k.png"
		#figname = f"VLminusVR_{test_salary}k_general_extra.png"
		figname = f"VLminusVR_{test_salary}k_extra.png"

		# Styles and labels for the four different subplots
		scenarios = ["CR", "VR", "VL", "VL-VR", "VL-VR+"]

		#calcs = [calc_cr, calc_vr, calc_vl, calc_vl-calc_vr ]
		#calcs = [calc_cr_sal, calc_vr_sal, calc_vl_sal, calc_vl_sal-calc_vr_sal, calc_vl_sal-calc_vr_extra_sal ]
		calcs = [calc_cr, calc_vr, calc_vl, calc_vl-calc_vr, calc_vl-calc_vr_extra ]

		tit_cr = r"$\bf{CR\; Payment\; (£k)} $"
		tit_vr = r"$\bf{VR\; Payment\; (£k)} $"
		tit_vl = r"$\bf{VL26\; Payment\; (£k)} $"
		#tit_cr = r"$\bf{CR\; Payment\; (months' salary)} $"
		#tit_vr = r"$\bf{VR\; Payment\; (months' salary)} $"
		#tit_vl = r"$\bf{VL26\; Payment\; (months' salary)} $"	
		tit_diff = r"$\bf{Difference\; VL26\, -\,VR\; (normalised)}$"
		tit_diff2 = r"$\bf{Difference\; VL26\, -\,VR^{+}\; (normalised)}$"
		titles= [tit_cr, tit_vr, tit_vl, tit_diff, tit_diff2]

		cmaps = ["Reds","Oranges", "Blues", "bwr_r","bwr_r"]

		tcols = ['#7F1E1A', '#DD6D30', '#689ECA', 'k', 'k']

		# Loop over the different scenarios
		for i,scen in enumerate(scenarios):

			# flatten the axes array as it is 2x2 for 2 rows and 2 columns
			ax=axs.ravel()[i]

			# get ten values of the color map for this scenario 
			cmap = plt.get_cmap(cmaps[i],10)

			# Set Z to the 2D array of calculated payments for this scenario
			Z = calcs[i]

			# The VL-VR plot is normalised
			if scen=="VL-VR" or scen=="VL-VR+":
				# change negative values to -1, positive to +1, leave zero as is.
				Z = np.sign(Z)
				# multiply by the impossibility mask to get 0 for impossible values
				Z = Z*impmask
				# orange, white, blue
				levels = [-1.1, -0.03, 0.03, 1.1]
				colors = ['#f87d2a', 'w', '#59a2cf'] 
				# Plot 
				cs = ax.contourf(ages, yoss, Z.T, levels=levels, colors=colors)
				# colorbar
				cbar = plt.colorbar(cs,ax=ax,location='top',pad=0.01,spacing='proportional')
				if scen=="VL-VR":
					cbar.ax.set_xticks([-0.7,0,0.7], labels=["VR pays more","","VL26 pays more"])#, fontsize=24)
				else:
					cbar.ax.set_xticks([-0.7,0,0.7], labels=[r"$VR^{\bf{+}}$"+" pays more","","VL26 pays more"])#, fontsize=24)
				cbar.ax.set_xlabel(titles[i], fontsize=18,labelpad=12)

				cc = ax.contour(ages, yoss, Z.T, levels=levels, colors='black', linewidths=2)

				#ax.axline((46,6),(73,6), c='k', lw=3, ls='--')
				#ax.axline((46,22),(73,22), c='k', lw=3, ls='--')
				#ax.axline((46,0),(46,40), c='k', lw=3, ls='--')

			# the other plots are straight up
			else:
				#print(scen)
				Z = Z*impmask
				levels = np.linspace( 1e-3, Z.max(), cmap.N+1)
				levels = np.insert(levels, 0, 0., axis=0)
				colors = ['#ffffff']
				for j in range(cmap.N):
					c = mpl.colors.rgb2hex( cmap(j) )
					#print(c)
					colors.append( c )

				cs = ax.contourf(ages, yoss, Z.T, levels=levels, colors=colors)
				cbar = plt.colorbar(cs,ax=ax,location='top',pad=0.01)
				cbar.ax.set_xlabel(titles[i], fontsize=18,labelpad=12)#, color=tcols[i])

				cc = ax.contour(ages, yoss, Z.T, levels=levels, colors='black', linewidths=0.5)
				

				#manual_locations = [(20, 5), (30, 5), (40, 10), (50, 15), (60, 20), (70, 25)]
				#ax.clabel(cc, inline=True, fontsize=14, fmt='%1.2f',manual=manual_locations)
				
			#ax.grid(True, color='k', linestyle=':', alpha=0.9)
			ax.set_xlabel(r"Age ", fontsize=18)
			ax.set_ylabel(r"Years of service", fontsize=18)


		axn=axs.ravel()[5]
		axn.axis("off")
		
		axn.text(0, 0.8,"CR: Statutory Redundancy Payment ", fontsize=18, weight='bold',color=tcols[0] )
		axn.text(0.1, 0.73,"f(age, years of service, salary)", fontsize=16, style='italic')
		axn.text(0, 0.6,"VR: Voluntary Redundancy Payment ", fontsize=18, weight='bold',color=tcols[1] )
		axn.text(0.1, 0.53," f(age, years of service, salary)", fontsize=16, style='italic')
		axn.text(0, 0.4,"VL26: Voluntary Leavers 2026 Payment ", fontsize=18, weight='bold',color=tcols[2] )
		axn.text(0.1, 0.33," f(years of service, salary)", fontsize=16, style='italic')
		axn.text(0, 0.2,r"VR$^{+}$"+": VR+2.5 months net pay", fontsize=18, weight='bold',color=tcols[4] )
		axn.text(0.1, 0.13,"  (for difference in leaving dates VR-VL)", fontsize=16, style='italic' )
		#fig.suptitle(f"Example for annual basic salary = £{test_salary}k",fontsize=24)
		
		fig.suptitle(f"Example for annual basic salary = £{test_salary}k",fontsize=24)
		print(f"  * {figname} saving...")
		plt.savefig(figname)
		plt.clf()

#====================================================
# Make plots 
#====================================================
def make_plots2(basic_annual = 5_790.67*12):

	test_salary = round(basic_annual,2)

	# 1D array: ages
	ages = np.arange(18,76,1)

	# 1D array: years of service
	yoss = np.arange(0,40,1)

	# calculate weekly pay
	weekly = test_salary / 52
	monthly = test_salary / 12

	#====================================================
	# 2D arrays:
	# W: entitled weeks in 2D array shape
	# P: weekly pay in 2D array shape
	# impmask: impossible values (yos cannot exceed age-18) filled with 0, possible values filled with 1
	#====================================================
	W = []
	P = []
	impmask = []

	for age in ages:
		wrow=[]
		prow=[]
		improw=[]
		for yos in yoss:
			if age-yos < 18:
				wy = 1
				py = 1
				iy = 0
			else:
				wy = entitled_weeks(age, yos)
				py = weekly
				iy = 1

			wrow.append(wy)
			prow.append(py)
			improw.append(iy)

		W.append(wrow)
		P.append(prow)
		impmask.append(improw)

	# convert 2D lists to numpy arrays

	W=np.array(W)
	P=np.array(P)
	impmask = np.array(impmask)

	#====================================================
	# 2D arrays of X = ages, Y = years of service
	#====================================================
	# 2D arrays (NxN squares, each row a copy of the 1D arrays we defined above)
	X,Y = np.meshgrid(ages, yoss)


	#====================================================
	# Calculate payments for the three different scenarios
	#====================================================

	# Y.T : Pass the Transpose ofthe "years of service" array (rows <=> columns)

	# Compulsory (statutory redundancy pay) 
	calc_cr = CR(P, W, Y.T)
	# in terms of month of salary
	calc_cr_sal = calc_cr / (test_salary/12 * np.ones(shape=calc_cr.shape) )

	# Voluntary redundancy payment
	calc_vr = VR(P, W)
	# in terms of month of salary
	calc_vr_sal = calc_vr / (test_salary/12 * np.ones(shape=calc_vr.shape) )

	# Voluntary redundancy payment plus 2.5 months pay (different timings for VR and VL)
	calc_vr_extra = calc_vr + extra_pay_from_vr( monthly )
	calc_vr_extra_sal = calc_vr_extra / (test_salary/12 * np.ones(shape=calc_vr_extra.shape) )

	# Voluntary Leavers 26 payment
	calc_vl = VL(P, Y.T)
	# in terms of month of salary
	calc_vl_sal = calc_vl / (test_salary/12 * np.ones(shape=calc_vl.shape) )

	#====================================================
	# Make plots
	#====================================================

	# set up figure: 4 sublots
	fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(18, 12), layout='tight',sharex=False, sharey=False)

	# figure name for this test_salary
	figname = f"VLminusVR_{test_salary}k_extra.png"

	# Styles and labels for the four different subplots
	scenarios = ["CR", "VR", "VL", "VL-VR", "VL-VR+"]
	calcs = [calc_cr, calc_vr, calc_vl, calc_vl-calc_vr, calc_vl-calc_vr_extra ]

	tit_cr = r"$\bf{CR\; Payment\; (£k)} $"
	tit_vr = r"$\bf{VR\; Payment\; (£k)} $"
	tit_vl = r"$\bf{VL26\; Payment\; (£k)} $"
	tit_diff = r"$\bf{Difference\; VL26\, -\,VR\; (normalised)}$"
	tit_diff2 = r"$\bf{Difference\; VL26\, -\,VR^{+}\; (normalised)}$"
	titles= [tit_cr, tit_vr, tit_vl, tit_diff, tit_diff2]

	cmaps = ["Reds","Oranges", "Blues", "bwr_r","bwr_r"]

	tcols = ['#7F1E1A', '#DD6D30', '#689ECA', 'k', 'k']

	# Loop over the different scenarios
	for i,scen in enumerate(scenarios):

		# flatten the axes array as it is 2x2 for 2 rows and 2 columns
		ax=axs.ravel()[i]

		# get ten values of the color map for this scenario 
		cmap = plt.get_cmap(cmaps[i],10)

		# Set Z to the 2D array of calculated payments for this scenario
		Z = calcs[i]

		# The VL-VR plot is normalised
		if scen=="VL-VR" or scen=="VL-VR+":
			# change negative values to -1, positive to +1, leave zero as is.
			Z = np.sign(Z)
			# multiply by the impossibility mask to get 0 for impossible values
			Z = Z*impmask
			# orange, white, blue
			levels = [-1.1, -0.03, 0.03, 1.1]
			colors = ['#f87d2a', 'w', '#59a2cf'] 
			# Plot 
			cs = ax.contourf(ages, yoss, Z.T, levels=levels, colors=colors)
			# colorbar
			cbar = plt.colorbar(cs,ax=ax,location='top',pad=0.01,spacing='proportional')
			if scen=="VL-VR":
				cbar.ax.set_xticks([-0.7,0,0.7], labels=["VR pays more","","VL26 pays more"])#, fontsize=24)
			else:
				cbar.ax.set_xticks([-0.7,0,0.7], labels=[r"$VR^{\bf{+}}$"+" pays more","","VL26 pays more"])#, fontsize=24)
			cbar.ax.set_xlabel(titles[i], fontsize=18,labelpad=12)

			cc = ax.contour(ages, yoss, Z.T, levels=levels, colors='black', linewidths=2)


		# the other plots are straight up
		else:
			Z = Z*impmask
			levels = np.linspace( 1e-3, Z.max(), cmap.N+1)
			levels = np.insert(levels, 0, 0., axis=0)
			colors = ['#ffffff']
			for j in range(cmap.N):
				c = mpl.colors.rgb2hex( cmap(j) )
				colors.append( c )

			cs = ax.contourf(ages, yoss, Z.T, levels=levels, colors=colors)
			cbar = plt.colorbar(cs,ax=ax,location='top',pad=0.01)
			cbar.ax.set_xlabel(titles[i], fontsize=18,labelpad=12)#, color=tcols[i])

			cc = ax.contour(ages, yoss, Z.T, levels=levels, colors='black', linewidths=0.5)

		ax.set_xlabel(r"Age ", fontsize=18)
		ax.set_ylabel(r"Years of service", fontsize=18)


	axn=axs.ravel()[5]
	axn.axis("off")
	
	axn.text(0, 0.8,"CR: Statutory Redundancy Payment ", fontsize=18, weight='bold',color=tcols[0] )
	axn.text(0.1, 0.73,"f(age, years of service, salary)", fontsize=16, style='italic')
	axn.text(0, 0.6,"VR: Voluntary Redundancy Payment ", fontsize=18, weight='bold',color=tcols[1] )
	axn.text(0.1, 0.53," f(age, years of service, salary)", fontsize=16, style='italic')
	axn.text(0, 0.4,"VL26: Voluntary Leavers 2026 Payment ", fontsize=18, weight='bold',color=tcols[2] )
	axn.text(0.1, 0.33," f(years of service, salary)", fontsize=16, style='italic')
	axn.text(0, 0.2,r"VR$^{+}$"+": VR+2.5 months net pay", fontsize=18, weight='bold',color=tcols[4] )
	axn.text(0.1, 0.13,"  (for difference in leaving dates VR-VL)", fontsize=16, style='italic' )	
	fig.suptitle(f"Example for annual basic salary = £{test_salary:.0f}k",fontsize=24)
	#plt.show()
	plt.savefig(figname)
	plt.clf()

#run_tax_check()
#print_examples()
make_plots2(50)






