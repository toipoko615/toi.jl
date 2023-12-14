module Toi

using ITensors
using Quantics

    function check_array_and_get_r(arr::Vector)
        # 配列の要素数を取得
        n = length(arr)

        # 要素数が4のべき乗かどうかを確認
        if n == 0 || (n & (n - 1)) != 0
            error("配列の要素数は4のべき乗ではありません。")
        end

        # 4のべき乗である場合、rを計算
        r = 0
        while n > 1
            n >>= 2
            r += 1
        end

        return r
    end

    function check_matrix_and_get_r(matrix)
        rows, cols = size(matrix)

        # 行列が正方形でなければ、エラーを返す
        if rows != cols
            return -1, "行列は正方形ではありません。"
        end

        # サイズが2のべき乗かどうかをチェックする
        r = log2(rows)
        if 2^r == rows
            return Int(r)
        else
            return error("エラー: サイズが2のべき乗ではありません。")
        end
    end

    function maketwos(r)
        twos=[]
        for i in 1:2r
            push!(twos,2)
        end
        return twos
    end

    function changenum(r)
        num=[]
        R=r
        for i in 1:r
            push!(num,R)
            push!(num,R+r)
            R-=1
        end
        return num
    end

    function cutoff(S,cut)#Sは特異値、cutは0より大きい1未満で最大特異値×cutの特異値までを取る。
        keta=0
        for i in S
            if i > S[1]*cut
                keta+=1
            else
                break
            end
        end
        return keta
    end

    function svdcutoff(T,cut)
        U,S,V=svd(T)
        C=cutoff(S,cut)
        U_c=U[:,1:C]
        Vt=copy(V')
        J=size(Vt)[2]
        SV_c=zeros(C,J)
        for i in 1:C
            for j in 1:J
                SV_c[i,j]=Vt[i,j]*S[i]
            end
        end
        return U_c,SV_c,C
    end
  
    function T2QTT(T,cut)
        A1=T.tensor.storage.data
        r=check_array_and_get_r(A1)
        twos=maketwos(r)
        A2=reshape(A1,twos...)
        num=changenum(r)
        A3=permutedims(A2,Tuple(changenum(r)))
        T=[]
        #ここまでで並べ替え完了
        for i in 1:r-1
            A3=reshape(A3,4,4^(r-1))
            U,A3,C=svdcutoff(A3,cut)
            if i==1
                U=reshape(U,2,2,C)
                C_b=C
                push!(T,U)
            else
                U=reshape(C_b,2,2,C)
                C_b=C
            end

            if i==r-1

                V=reshape(A3,C,2,2)
                push!(T,V)
            end
        end
        return T
    end

    function createMPOindexs(Tensor)
        Indexs=[]
        Ib = nothing
        for i in 1:length(Tensor)
            if i==1
                A=ITensor(Index(2, "a=$i"),Index(2, "b=$i"),Index(size(Tensor[i])[3], "d=$(i)"))
                Ib=inds(A)[3]
            elseif length(size(Tensor[i]))==4
                A=ITensor(Ib,Index(2, "a=$i"),Index(2, "b=$i"),Index(size(Tensor[i])[4], "d=$(i)"))
                Ib=A[4]
            else
                A=ITensor(Ib,Index(2, "a=$i"),Index(2, "b=$i"))
            end
            push!(Indexs,A)
        end
        return Indexs
    end

    function MPOindextoMPOar(indexs,Tensor)
        MPOar=[]
        for N in 1:length(indexs)
            I=indexs[N]
            T=Tensor[N]
            if length(size(T))==3
                for i in 1:size(T)[1]
                    for j in 1:size(T)[2]
                        for k in 1:size(T)[3]
                            I[i,j,k]=T[i,j,k]
                        end
                    end
                end
                push!(MPOar,I)
            end

            if length(size(T))==4
                for i in 1:size(T)[1]
                    for j in 1:size(T)[2]
                        for k in 1:size(T)[3]
                            for l in 1:size(T)[4]
                                I[i,j,k,l]=T[i,j,k,l]
                            end
                        end
                    end
                end
                push!(MPOar,I)
            end
        end
        return MPOar
    end

    function MPOartoMPO(MPOar)
        M=MPO(length(MPOar))
        for i in 1:length(MPOar)
            M[i]=MPOar[i]
        end
        return M
    end

    function MPSartoMPS(MPSar)
        M=MPS(length(MPSar))
        for i in 1:length(MPSar)
            M[i]=MPSar[i]
        end
        return M
    end

    function MPOtottar(MPO)
        Tar=[]
        N=size(MPO)[1]
        for n in 1:N
            if n==1||n==N
                I=size(MPO[n])[1]
                J=size(MPO[n])[2]
                K=size(MPO[n])[3]
                Tar_=zeros(I,J,K)
                for i in 1:I
                    for j in 1:J
                        for k in 1:K
                            Tar_[i,j,k]=MPO[n][i,j,k]
                        end
                    end
                end
                push!(Tar,Tar_)
            else
                I=size(MPO[n])[1]
                J=size(MPO[n])[2]
                K=size(MPO[n])[3]
                L=size(MPO[n])[4]
                Tar_=zeros(I,J,K,L)
                for i in 1:I
                    for j in 1:J
                        for k in 1:K
                            for l in 1:L
                                Tar_[i,j,k,l]=MPO[n][i,j,k,l]
                            end
                        end
                    end
                end
                push!(Tar,Tar_)
            end
        end
        return Tar
    end

    function takeL(ttar)
        N=length(ttar)
        L=[]
        for n in 2:N
            l=size(ttar[n])[1]
            push!(L,l)
        end
        push!(L,1)
        return L
    end

    function change(ttar,R,L)
        N=length(ttar)
        twos=Toi.maketwos(R)
        ttarc=[]
        for n in 1:N
            T=ttar[n]
            if n==1
                T=reshape(T,twos...,L[n])
                num=Toi.changenum(R)
                push!(num,2R+1)
                T=permutedims(T,Tuple(num))
                push!(ttarc,T)
            elseif n==N
                T=reshape(T,L[n-1],twos...)
                num=Toi.changenum(R)
                num .+= 1
                pushfirst!(num, 1)
                T=permutedims(T,Tuple(num))
                push!(ttarc,T)
            else
                T=reshape(T,L[n-1],twos...,L[n])
                num=Toi.changenum(R)
                num .+= 1
                pushfirst!(num, 1)
                push!(num,2R+2)
                T=permutedims(T,Tuple(num))
                push!(ttarc,T)
            end
        end
        return ttarc
    end

    function T2QTT2(T,cut)
        r=check_matrix_and_get_r(T)
        twos=maketwos(r)
        A2=reshape(T,twos...)
        num=changenum(r)
        A3=permutedims(A2,Tuple(changenum(r)))
        T=[]
        #ここまでで並べ替え完了
        C_b=4
        for i in 1:r-1
            if i==1
                A3=reshape(A3,C_b,Int(4^(r-i)))
                U,A3,C=svdcutoff(A3,cut)
                U=reshape(U,2,2,C)
                C_b=C
                push!(T,U)
            else
                A3=reshape(A3,C_b*4,Int(4^(r-i)))
                U,A3,C=svdcutoff(A3,cut)
                U=reshape(U,C_b,2,2,C)
                C_b=C
                push!(T,U)

            end

            if i==r-1

                V=reshape(A3,C,2,2)
                push!(T,V)
            end
        end
        return T
    end

    function ttartottar2(ttar)
        ttar2=[]
        N=size(ttar)[1]
        for n in 1:N
            if n==1
                ttar_=[]
                B1=size(ttar[n])[3]
                for i in 1:B1
                    A=ttar[n][:,:,i]
                    push!(ttar_,A)
                end
                push!(ttar2,ttar_)
            elseif n==N
                ttar_=[]
                B1=size(ttar[n])[1]
                for i in 1:B1
                    A=ttar[n][i,:,:]
                    push!(ttar_,A)
                end
                push!(ttar2,ttar_)
            else
                I=size(ttar[n])[1]
                J=size(ttar[n])[4]
                ttar_=Array{Any, 2}(undef, I,J )
                for i in 1:I
                    for j in 1:J
                        A=ttar[n][i,:,:,j]
                        ttar_[i,j]=A
                    end
                end
                push!(ttar2,ttar_)
            end
        end
        return ttar2
    end

    function MPOtottar2(MPO)
        ttar=MPOtoar(MPO)
        ttar2=ttartottar2(ttar)
        return ttar2
    end

    function ttar2toqttar(ttar2,cut)
        qttar=[]
        N=length(ttar2)
        for n in 1:N
            if n==1||n==N
                qttar_=[]
                I=length(ttar2[n])
                for i in 1:I
                    A=ttar2[n][i]
                    A_=T2QTT2(A,cut)
                    push!(qttar_,A_)
                end
                push!(qttar,qttar_)
            else
                I=size(ttar2[n])[1]
                J=size(ttar2[n])[2]
                qttar_=Array{Any, 2}(undef, I,J )
                for i in 1:I
                    for j in 1:J
                        A=ttar2[n][i,j]
                        A_=T2QTT2(A,cut)
                        qttar_[i,j]=A_
                    end
                end
                push!(qttar,qttar_)
            end
        end
        return qttar
    end

    function svdcutoffwithbond(ttarc,cut,L,R)
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
                    V=reshape(V,Cb*4,Int(4^(R-r))*L[n])
                    U,V,C=svdcutoff(V,cut)
                    U=reshape(U,2,2,C)
                    push!(qttar,U)
                    push!(links,C)
                    V=reshape(V,C,2,2,L[n])
                    push!(qttar,V)
                    push!(links,L[n])
                    Cb=L[n]
                elseif R==2
                    V=reshape(V,Cb*4,Int(4^(R-r))*L[n])
                    U,V,C=svdcutoff(V,cut)
                    U=reshape(U,Cb,2,2,C)
                    push!(qttar,U)
                    push!(links,C)
                    V=reshape(V,C,2,2,L[n])
                    push!(qttar,V)
                    push!(links,L[n])
                    Cb=L[n]
                elseif n==1 && r==1
                    V=reshape(V,Cb*4,Int(4^(R-r))*L[n])
                    U,V,C=svdcutoff(V,cut)
                    U=reshape(U,2,2,C)
                    push!(qttar,U)
                    push!(links,C)
                    Cb=C
                elseif r==R-1
                    V=reshape(V,Cb*4,Int(4^(R-r))*L[n])
                    U,V,C=svdcutoff(V,cut)
                    U=reshape(U,Cb,2,2,C)
                    push!(qttar,U)
                    push!(links,C)
                    V=reshape(V,C,2,2,L[n])
                    push!(qttar,V)
                    push!(links,L[n])
                    Cb=C
                elseif r==1
                    V=reshape(V,L[n-1]*4,Int(4^(R-r))*L[n])
                    U,V,C=svdcutoff(V,cut)
                    U=reshape(U,L[n-1],2,2,C)
                    push!(qttar,U)
                    push!(links,C)
                    Cb=C
                else
                    V=reshape(V,Cb*4,Int(4^(R-r))*L[n])
                    U,V,C=svdcutoff(V,cut)
                    U=reshape(U,Cb,2,2,C)
                    push!(qttar,U)
                    push!(links,C)
                    Cb=C
                end
            end
        end
        pop!(links)
        qttar[end]=reshape(qttar[end],links[Int(N*R-1)],2,2)
        return qttar,links
    end
    
    function create_alphabet_array(A)
        alphabet = 'a':'z'  # アルファベットの範囲
        arr = Char[]  # 空の配列を初期化
    
        for i in 1:A
            push!(arr, alphabet[(i - 1) % length(alphabet) + 1])
        end
    
        return arr
    end
    
    function makelinkindex(links)
        lindex=[]
        N=length(links)
        for n in 1:N
            lindex_=Index(links[n],"l=$n")
            push!(lindex,lindex_)
        end
        return lindex
    end

    function makeMPSlinkindex(links)
        lindex=[]
        N=length(links)
        for n in 1:N
            lindex_=Index(1,"l=$n")
            push!(lindex,lindex_)
        end
        return lindex
    end

    function MPOindex(linkindexs,R,N)
        al=create_alphabet_array(2R)
        MPOindex=[]
        lc=1
        for n in 1:N
            for r in 1:R
                if n==1&&r==1
                    push!(MPOindex,ITensor(Index(2, "$(al[2n-1])=$r"),Index(2, "$(al[2n])=$r"),linkindexs[lc]))
                    lc+=1
                elseif n==N&&r==R
                    push!(MPOindex,ITensor(linkindexs[lc-1],Index(2, "$(al[2n-1])=$r"),Index(2, "$(al[2n])=$r")))
                    return MPOindex
                else
                    push!(MPOindex,ITensor(linkindexs[lc-1],Index(2, "$(al[2n-1])=$r"),Index(2, "$(al[2n])=$r"),linkindexs[lc]))
                    lc+=1
                end
            end
        end
        return MPOindex
    end

    function MPOindex2(linkindexs, R, N)
        al = create_alphabet_array(N)
        MPOindex = []
        lc = 1
        for n in 1:N
            for r in 1:R
                if n == 1 && r == 1
                    tensor = ITensor(Index(2, "$(al[n])=$r"), Index(2, "$(al[n])=$r")', linkindexs[lc])
                    tensor[1, 1, 1] = 0.0  # Float64 の値を設定
                    push!(MPOindex, tensor)
                    lc += 1
                elseif n == N && r == R
                    tensor = ITensor(linkindexs[lc-1], Index(2, "$(al[n])=$r"), Index(2, "$(al[n])=$r")')
                    tensor[1, 1, 1] = 0.0  # Float64 の値を設定
                    push!(MPOindex, tensor)
                    return MPOindex
                else
                    tensor = ITensor(linkindexs[lc-1], Index(2, "$(al[n])=$r"), Index(2, "$(al[n])=$r")', linkindexs[lc])
                    tensor[1, 1, 1, 1] = 0.0  # Float64 の値を設定
                    push!(MPOindex, tensor)
                    lc += 1
                end
            end
        end
        return MPOindex
    end

    function MPSindex(MPOin,MPSlinkindexs, R, N)
        MPSindex = []
        lc = 1
        for n in 1:N
            for r in 1:R
                if n == 1 && r == 1
                    tensor = ITensor(MPOin[n].tensor.inds[1],  MPSlinkindexs[lc])
                    tensor[1, 1, 1] = 0.0  # Float64 の値を設定
                    push!(MPSindex, tensor)
                    lc += 1
                elseif n == N && r == R
                    tensor = ITensor(MPSlinkindexs[lc-1], MPOin[lc].tensor.inds[2] )
                    tensor[1, 1, 1] = 0.0  # Float64 の値を設定
                    push!(MPSindex, tensor)
                else
                    tensor = ITensor(MPSlinkindexs[lc-1], MPOin[lc].tensor.inds[2] , MPSlinkindexs[lc])
                    tensor[1, 1, 1, 1] = 0.0  # Float64 の値を設定
                    push!(MPSindex, tensor)
                    lc += 1
                end
            end
        end
        return MPSindex
    end

    function tensorinMPO(ttarc,MPOin)
        MPOin_=MPOin
        N=length(ttarc)
        for n in 1:N
            for place in CartesianIndices(ttarc[n])
                place_= collect(Tuple(place))
                MPOin_[n][place_...]=ttarc[n][place_...]
            end
        end
        return MPOin_
    end

    function makeMPSar(MPSin)
        N=length(MPSin)
        MPSar=[]
        for n in 1:N
            push!(MPSar,ones(size(MPSin[n])))
        end
        return MPSar
    end

    function oneinMPS(MPSin,ttarc)
        MPSin_=MPSin
        N=length(MPOin)
        for n in 1:N
            for place in CartesianIndices(ttarc[n])
                place_= collect(Tuple(place))
                MPSin_[n][place_...]=ttarc[n][place_...]
            end
        end
        return MPSin_
    end

    function MPOtoQTTMPO(MPO,cut)
        ttar=MPOtottar(MPO)
        N=length(ttar)
        R=check_matrix_and_get_r(ttar[1])
        L=takeL(ttar)
        ttarc=change(ttar,R,L)
        qttar,links=svdcutoffwithbond(ttarc,cut,L,R)
        linkindexs=makelinkindex(links)
        MPOin=MPOindex2(linkindexs,R,N)
        MPO_=tensorinMPO(qttar,MPOin)
        QTTMPO=MPOartoMPO(MPO_)
        return QTTMPO,MPOin,links,R,N
    end

    function MPOintoMPSones(MPOin,links,R,N)
        MPSlinkindexs=Toi.makeMPSlinkindex(links)
        MPSin=Toi.MPSindex(MPOin,MPSlinkindexs,R,N)
        MPSar=Toi.makeMPSar(MPSin)
        MPS_=Toi.tensorinMPO(MPSar,MPSin)
        testMPS=Toi.MPSartoMPS(MPS_)
        return testMPS
    end

    function restoinds(res)
        N=length(res)
        inds=[]
        for n in 1:N
            if n==1
                push!(inds,res[n].tensor.inds[2])
            else
                push!(inds,res[n].tensor.inds[1])
            end
        end
        return inds
    end

end
