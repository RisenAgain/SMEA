

def MSR_RV_BI(Bicluster,P,Q,MSR_threshold):

    cal = []  # store the value of squared part for MSR
    row = []  # store the value of squared part for row variance
    for i in range(0, P):
        for j in range(0, Q):

            # element of bicluster at i th row,j th column
            a = Bicluster[i, j]
            #print("a:", a)

            # Mean of i th row
            temp = 0
            for x in range(0, Q):
                temp += Bicluster[i, x]
            b = (1 / Q) * temp
            #print("b:", b)

            # Mean of j th column
            temp = 0
            for x in range(0, P):
                temp += Bicluster[x,j]
            c = (1 / P) * temp
            #print("c:", c)

            # Mean of all the elements in the bicluster
            temp = 0
            for x in range(0, P):
                for y in range(0, Q):
                    temp += Bicluster[x, y]
            d = (1 / (P * Q)) * temp
            #print("d:", d)
            v_cal = (a + d - (b + c)) ** 2  # calculation of summation for each element of each bicluster
            cal.append(v_cal)
            v_row = (a - b) ** 2
            row.append(v_row)
    #print(cal)
    #print(row)
    MSR_value = (float(1) / (P * Q)) * sum(cal)
    #print("MSR value:", MSR_value)
    if MSR_value >= MSR_threshold:  # if MSR value is less than threshold value then it is added to objective1 list else ignored
        MSR_value_lessthan_threshold = MSR_value
        Row_variance_value = (float(1) / (P * Q)) * sum(row)
        #print("Row variance value:", Row_variance_value)
        BI = (MSR_value) / (1 + Row_variance_value)
        #print("Bicluster index value:", BI)
        return MSR_value_lessthan_threshold,Row_variance_value,BI
    else:
        return 0,0,0
