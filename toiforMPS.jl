module Toimps

    using ITensors
    using Quantics
    include("toi.jl")
    using .Toi

    function createMPSTensors(N, D, d)
        tensors = Array{Float64}[]
        for i in 1:N
            if i == 1
                push!(tensors, rand(d,  D))
            elseif i == N
                push!(tensors, rand(D, d))
            else
                push!(tensors, rand(D, d, D))
            end
        end
        return tensors
    end

    function createITTensors(N, D, d)
        # サイトとボンドのインデックスを生成
        q = [Index(d, "Species,q=$i") for i in 1:N] # ローカル（物理）インデックス
        l = [Index(D, "Link,l=$i") for i in 1:N-1] # ボンドインデックス、l[1] と l[N+1] は両端

        # ITensorの配列を初期化
        tensors = ITensor[]
        for i in 1:N
            if i == 1
                tensor = ITensor(q[i], l[i])
            elseif i == N
                tensor = ITensor(l[i-1], q[i])
            else
                tensor = ITensor(l[i-1], q[i], l[i])
            end
            
            push!(tensors, tensor)
        end
        return tensors,q
    end

    function processTensorIndices(TT_1,T)
        TT_1c=copy(TT_1)
        N=length(TT_1)
        for n in 1:N
            sizes = size(TT_1[n])
            ranges = [1:s for s in sizes]
        
            for inds in Iterators.product(ranges...)
                TT_1c[n][inds...]=T[n][inds...]
            end
        end

        return TT_1c

    end

    function toMPS(TT_2,N)
        TT=MPS(N)
        for n in 1:N
            TT[n]=TT_2[n]
        end
        return TT
    end

    function randMPS(N,D,d)#ランダムなMPSを生成する関数
        TT_1,q=createITTensors(N, D, d)
        T=createMPSTensors(N, D, d)
        TT_2=processTensorIndices(TT_1,T)
        TT=toMPS(TT_2,N)
        return TT,q
    end

    function linklocallink(TT)
        N=length(TT)
        TTch=[]
        for n in 1:N
            if n==1
                index1 = findindex(TT[n], "Species,q=$n")
                index2 = findindex(TT[n], "Link,l=$n")
                push!(TTch,permute(TT[n], (index1, index2)))
            elseif n==N
                index1 = findindex(TT[n], "Species,q=$n")
                index2 = findindex(TT[n], "Link,l=$(n-1)")
                push!(TTch,permute(TT[n], (index2, index1)))
            else
                index1 = findindex(TT[n], "Species,q=$n")
                index2 = findindex(TT[n], "Link,l=$n")
                index3 = findindex(TT[n], "Link,l=$(n-1)")
                push!(TTch,permute(TT[n], (index3,index1, index2)))
            end
        end
        return TTch
    end

    function TTtoTTar(TT)
        N=length(TT)
        TTar=[]
        for n in 1:N
            dims = size(TT[n]) 
            TTar_=zeros(dims...)
            for indices in Base.Iterators.product((1:n for n in dims)...)
                TTar_[indices...]=TT[n][indices...]
            end
            push!(TTar,TTar_)
        end
        return TTar
    end

    function takeL(TTar)
        N=length(TTar)
        L=[]
        for n in 2:N
            push!(L,size(TTar[n])[1])
        end
        push!(L,1)
        return L
    end

    function getR(TTar)
        N=size(TTar[1])[1]
        # Check if N is a positive integer
        if N <= 0 || !isa(N, Int)
            error("N must be a positive integer")
        end
    
        # Check if N is a power of 2
        if (N & (N - 1)) == 0
            # Calculate R using bit manipulation
            R = 0
            while N > 1
                N >>= 1
                R += 1
            end
            return R
        else
            error("N is not a power of 2")
        end
    end

    function makenum(R)
        num=[]
        for r in 1:R
            push!(num,R+1-r)
        end
        return num
    end

    function maketwos(R)
        twos=[]
        for r in 1:R
            push!(twos,2)
        end
        return twos
    end

    function changetensor(TTar,R,L)
        N=length(TTar)
        num=makenum(R)
        twos=maketwos(R)
        TTarch=[]
        for n in 1:N
            if n==1
                num_=[num...,R+1]
                T=reshape(TTar[n],twos...,L[n])
                push!(TTarch,permutedims(T,num_))
            elseif n==N
                num_=num.+1
                num_=[1,num_...]
                T=reshape(TTar[n],L[n-1],twos...)
                push!(TTarch,permutedims(T,num_))
            else
                num_=num.+1
                num_=[1,num_...,R+2]
                T=reshape(TTar[n],L[n-1],twos...,L[n])
                push!(TTarch,permutedims(T,num_))
            end
        end
        return TTarch
    end

    function MPSsvdcutoffwithbond(ttarc,cut,L,R)
        N=length(ttarc)
        qttar=[]
        links=[]
        lb=1
        V=nothing
        Cb=1
        for n in 1:N
            V=ttarc[n]
            for r in 1:R-1
                if R==2 && n==1
                    V=reshape(V,Cb*2,Int(2^(R-r))*L[n])
                    U,V,C=Toi.svdcutoff(V,cut)
                    U=reshape(U,2,C)
                    push!(qttar,U)
                    push!(links,C)
                    V=reshape(V,C,2,L[n])
                    push!(qttar,V)
                    push!(links,L[n])
                    Cb=L[n]
                elseif R==2
                    V=reshape(V,Cb*2,Int(2^(R-r))*L[n])
                    U,V,C=Toi.svdcutoff(V,cut)
                    U=reshape(U,Cb,2,C)
                    push!(qttar,U)
                    push!(links,C)
                    V=reshape(V,C,2,L[n])
                    push!(qttar,V)
                    push!(links,L[n])
                    Cb=L[n]
                elseif n==1 && r==1
                    V=reshape(V,Cb*2,Int(2^(R-r))*L[n])
                    U,V,C=Toi.svdcutoff(V,cut)
                    U=reshape(U,2,C)
                    push!(qttar,U)
                    push!(links,C)
                    Cb=C
                elseif r==R-1
                    V=reshape(V,Cb*2,Int(2^(R-r))*L[n])
                    U,V,C=Toi.svdcutoff(V,cut)
                    U=reshape(U,Cb,2,C)
                    push!(qttar,U)
                    push!(links,C)
                    V=reshape(V,C,2,L[n])
                    push!(qttar,V)
                    push!(links,L[n])
                    Cb=C
                elseif r==1
                    V=reshape(V,L[n-1]*2,Int(2^(R-r))*L[n])
                    U,V,C=Toi.svdcutoff(V,cut)
                    U=reshape(U,L[n-1],2,C)
                    push!(qttar,U)
                    push!(links,C)
                    Cb=C
                else
                    V=reshape(V,Cb*2,Int(2^(R-r))*L[n])
                    U,V,C=Toi.svdcutoff(V,cut)
                    U=reshape(U,Cb,2,C)
                    push!(qttar,U)
                    push!(links,C)
                    Cb=C
                end
            end
        end
        pop!(links)
        qttar[end]=reshape(qttar[end],links[Int(N*R-1)],2)
        return qttar,links
    end

    function MPSindex(linkindexs, R, N)
        al = Toi.create_alphabet_array(N)
        MPSindex = []
        lc = 1
        for n in 1:N
            for r in 1:R
                if n == 1 && r == 1
                    tensor = ITensor(Index(2, "$(al[n])=$r"), linkindexs[lc])
                    tensor[1, 1, 1] = 0.0  # Float64 の値を設定
                    push!(MPSindex, tensor)
                    lc += 1
                elseif n == N && r == R
                    tensor = ITensor(linkindexs[lc-1], Index(2, "$(al[n])=$r"))
                    tensor[1, 1, 1] = 0.0  # Float64 の値を設定
                    push!(MPSindex, tensor)
                else
                    tensor = ITensor(linkindexs[lc-1], Index(2, "$(al[n])=$r"), linkindexs[lc])
                    tensor[1, 1, 1, 1] = 0.0  # Float64 の値を設定
                    push!(MPSindex, tensor)
                    lc += 1
                end
            end
        end
        return MPSindex
    end

    function MPStoQTTMPS(N,cut,TT)#MPSをQTTMPSに変換
        TTch=linklocallink(TT)
        TTar=TTtoTTar(TTch)
        L=takeL(TTar)
        R=getR(TTar)
        TTarch=changetensor(TTar,R,L)
        qttar,links=MPSsvdcutoffwithbond(TTarch,cut,L,R)
        linkindexs=Toi.makelinkindex(links)
        MPSin=MPSindex(linkindexs,R,N)
        MPS_=Toi.tensorinMPO(qttar,MPSin)
        QTTMPS=Toi.MPSartoMPS(MPS_)
        return QTTMPS,TTch,R
    end
end

module randmpo

    using ITensors
    using Quantics

    function createMPOTensors(N, D, d)
        tensors = Array{Float64}[]
        for i in 1:N
            if i == 1
                push!(tensors, rand(d, d,  D))
            elseif i == N
                push!(tensors, rand(D, d, d))
            else
                push!(tensors, rand(D, d, d, D))
            end
        end
        return tensors
    end

    function createMPOind(N, D, d, q)
        # サイトとボンドのインデックスを生成
        q2 = [Index(d, "Species,q=$i'") for i in 1:N] # ローカル（物理）インデックス
        l = [Index(D, "Link,l=$i") for i in 1:N-1] # ボンドインデックス、l[1] と l[N+1] は両端

        # ITensorの配列を初期化
        tensors = ITensor[]
        for i in 1:N
            if i == 1
                tensor = ITensor(q[i], q2[i], l[i])
            elseif i == N
                tensor = ITensor(l[i-1], q[i], q2[i])
            else
                tensor = ITensor(l[i-1], q[i], q2[i], l[i])
            end
            
            push!(tensors, tensor)
        end
        return tensors

    end

    function processTensorIndices(TT_1,T)
        TT_1c=copy(TT_1)
        N=length(TT_1)
        for n in 1:N
            sizes = size(TT_1[n])
            ranges = [1:s for s in sizes]
        
            for inds in Iterators.product(ranges...)
                TT_1c[n][inds...]=T[n][inds...]
            end
        end

        return TT_1c

    end

    function toMPO(TT_2,N)
        TT=MPO(N)
        for n in 1:N
            TT[n]=TT_2[n]
        end
        return TT
    end

    function randMPO(N,D,d,q)#ランダムなMPSを生成する関数
        TT_1=createMPOind(N, D, d, q)
        T=createMPOTensors(N, D, d)
        TT_2=processTensorIndices(TT_1,T)
        TT=toMPO(TT_2,N)
        return TT
    end

end

