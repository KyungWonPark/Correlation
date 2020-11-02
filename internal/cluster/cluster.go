package cluster

import (
	"math"
	"sort"

	"github.com/gonum/matrix/mat64"
)

// GetMultiplicity returns multiplicity of eigenvalue - Checked: Working
func GetMultiplicity(eigVal *mat64.Dense, gamma float64) int {
	multiplicity := -1
	n, _ := eigVal.Dims()
	// eigVal must be (n, 1) dimension matrix

	if gamma < 0.000000000001 {
		for i := 0; i < n; i++ {
			if i == (n - 1) {
				multiplicity = n
				break
			}

			valNow := math.Abs(eigVal.At(i, 0))
			valNext := math.Abs(eigVal.At(i+1, 0))

			isScaleDiff := false
			if valNext >= valNow*10e4 {
				isScaleDiff = true
			}

			isValNextBigEnough := false
			if valNext > 10e-7 {
				isValNextBigEnough = true
			}

			if isScaleDiff && isValNextBigEnough {
				multiplicity = i + 1
				break
			}
		}
	} else {
		targetVal := math.Abs(float64(n) * gamma)
		for i := 0; i < n; i++ {
			if i == (n - 1) {
				multiplicity = n
				break
			} else if i > 0 {
				valNow := math.Abs(eigVal.At(i, 0))
				diff := math.Abs(valNow - targetVal)

				if diff > 10e-5 {
					multiplicity = i
					break
				}
			}
		}
	}

	return multiplicity
}

// Cluster represents a Cluster
type Cluster struct {
	Identifier float64
	Members    []int
}

// AddMember adds a member to the cluster
func (c *Cluster) AddMember(idx int) {
	c.Members = append(c.Members, idx)
	return
}

func remove(s []int, i int) []int {
	s[i] = s[len(s)-1]
	return s[:len(s)-1]
}

// DelMember removes a member from the cluster
func (c *Cluster) DelMember(idx int) {
	for i := 0; i < len(c.Members); i++ {
		if c.Members[i] == idx {
			c.Members = remove(c.Members, i)
		}
	}
	return
}

type nodeEigVecElePair struct {
	nodeIdx    int
	vecEle     float64
	membership int
}

// GetClsFromOneEigVec returns cluster configuration from one eigen vector
func GetClsFromOneEigVec(eigVec *mat64.Dense, idx int) []Cluster {
	var clusters []Cluster

	_, cols := eigVec.Dims()
	sortedVec := make([]nodeEigVecElePair, cols)

	for i := 0; i < cols; i++ {
		sortedVec[i] = nodeEigVecElePair{
			nodeIdx:    i,
			vecEle:     eigVec.At(idx, i),
			membership: -1,
		}
	}

	// Sorting (in ascending order)
	sort.Slice(sortedVec, func(i, j int) bool {
		return sortedVec[i].vecEle < sortedVec[j].vecEle
	})

	// Inserting
	// i := 0
	c := Cluster{
		Identifier: sortedVec[0].vecEle,
		Members:    []int{},
	}
	sortedVec[0].membership = 0
	c.AddMember(sortedVec[0].nodeIdx)

	for i := 1; i < cols; i++ {
		if math.Abs(sortedVec[i].vecEle-sortedVec[i-1].vecEle) <= 3e-12 {
			sortedVec[i].membership = sortedVec[i-1].membership
			clusters[sortedVec[i].membership].AddMember(sortedVec[i].nodeIdx)
		} else {
			c := Cluster{
				Identifier: sortedVec[i].vecEle,
				Members:    []int{},
			}
			sortedVec[i].membership = len(clusters)
			c.AddMember(sortedVec[i].nodeIdx)

			clusters = append(clusters, c)
		}
	}

	return clusters
}

// GetEssentialClusters returns clusters with cluster size >= 2
func GetEssentialClusters(clusters []Cluster) []Cluster {
	var essClusters []Cluster

	for i := 0; i < len(clusters); i++ {
		if len(clusters[i].Members) >= 2 {
			essClusters = append(essClusters, clusters[i])
		}
	}

	sort.Slice(essClusters, func(i, j int) bool {
		return len(essClusters[i].Members) > len(essClusters[j].Members)
	})

	return essClusters
}

// GammaClustering returns cluster configuration, which returns only conf that has (M - 1) clusters, returns an empty slice otherwise
func GammaClustering(eigVal *mat64.Dense, eigVec *mat64.Dense, gamma float64) []Cluster {
	var answer []Cluster

	M := GetMultiplicity(eigVal, gamma)
	for i := 1; i < M; i++ {
		cls := GetClsFromOneEigVec(eigVec, i)
		if len(cls) == (M - 1) {
			answer = cls
			break
		}
	}

	return answer
}
