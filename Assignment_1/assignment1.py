def lowertriangular(n):
    for i in range(1,n+1):
        print("* " *i)

def uppertriangular(n):
    for i in range(1,n+1):
        print("* " * (n))
        n-=1
        
def pyramid(n):
    for i in range(1,n+1):
        print(" "*(n-1) + "* "*i )        #here we are multiplying space and decreasing it by 1
        n-=1
    
lowertriangular(6)
print("")
uppertriangular(6)
print("")
pyramid(6)