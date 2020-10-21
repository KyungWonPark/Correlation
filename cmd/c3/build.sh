#!/bin/bash

sed -i 's/^\/\/ "gonum.org/v1/gonum/blas/blas64"$/"gonum.org/v1/gonum/blas/blas64"/g' main.go
sed -i 's/^\/\/ blas_netlib "gonum.org/v1/netlib/blas/netlib"$/blas_netlib "gonum.org/v1/netlib/blas/netlib"/g' main.go
sed -i 's/^\/\/ blas64.Use(blas_netlib.Implementation{})$/blas64.Use(blas_netlib.Implementation{})/g' main.go

export CGO_LDFLAGS="-L$ROOT/opt/openblas/lib -lopenblas"
go build .