subroutine gauss(x, p, y, n, np)

    real, parameter :: pi=3.1415927
    integer :: n, np
    real, dimension(0:np-1), intent(in) :: p
    real, dimension(0:n-1), intent(in) :: x
    real, dimension(0:n-1), intent(out) :: y

    y(:) = 0

    do i=0, (np-1), 3
        y = y + p(i)*exp(-(x-p(i+1))**2/2./p(i+2)**2)
    enddo
    
end



subroutine gausshermite(x, p, y, n, np)

    real, parameter :: pi=3.1415927, sq2=sqrt(2.0), sq6=sqrt(6.0)
    real, parameter :: sq24=sqrt(24.0), sq2pi=sqrt(2.0*pi)
    integer :: n, np
    integer :: i
    real, dimension(0:np-1), intent(in) :: p
    real, dimension(0:n-1), intent(in) :: x
    real, dimension(0:n-1) :: w, hh3, hh4, alphag, gh
    real, dimension(0:n-1), intent(out) :: y
    real :: h3, h4
    real :: a, l0, s

    y(:) = 0.0
 
    do i=0, (np-1), 5

        a = p(i)
        l0 = p(i+1)
        s = p(i+2)
        h3 = p(i+3)
        h4 = p(i+4)

        w = (x-l0)/s

        alphag = 1./sq2pi*exp(-w**2/2.)
        hh3 = 1./sq6*(2.0*sq2*w**3-3.0*sq2*w)
        hh4 = 1./sq24*(4.0*w**4-12.0*w**2+3.0)

        gh = a*alphag/s*(1.0 + h3*hh3 + h4*hh4)

        y = y + gh

    enddo
    
end

subroutine gaussvel(x, rest_wl, p, y, n, np)

    real, parameter :: pi=3.1415927, c=299792.458
    integer :: n, np, j, i
    real, dimension(0:np-1), intent(in) :: p
    real, dimension(0:(np-1) / 3), intent(in) :: rest_wl
    real, dimension(0:n-1), intent(in) :: x
    real, dimension(0:n-1), intent(out) :: y
    real, dimension(0:n-1) :: w, vel, fvel, lam_ratio

    vel(:) = 0.0
    y(:) = 0.0

    j = 0
    do i=0, (np-1), 3
        lam_ratio = (x / rest_wl(j)) ** 2
        vel = c * (lam_ratio - 1.0) / (lam_ratio + 1.0)

        a = p(i)
        v0 = p(i+1)
        s = p(i+2)

        w = (vel - v0) / s
        ! The observed flux density equals the emitted flux density divided by (1 + z)
        fvel = a * exp(-w**2 / 2.0) / (1.0 + (vel / c))

        y = y + fvel 
        j = j + 1

    enddo
    
end

subroutine gausshermitevel(x, rest_wl, p, y, n, np)

    real, parameter :: pi=3.1415927, c=299792.458, sq2=sqrt(2.0), sq6=sqrt(6.0)
    real, parameter :: sq24=sqrt(24.0), sq2pi=sqrt(2.0*pi)
    integer :: n, np, i, j
    real, dimension(0:(np-1) / 5), intent(in) :: rest_wl
    real, dimension(0:np-1), intent(in) :: p
    real, dimension(0:n-1), intent(in) :: x
    real, dimension(0:n-1) :: w, hh3, hh4, alphag, vel, fvel, lam_ratio
    real, dimension(0:n-1), intent(out) :: y
    real :: h3, h4
    real :: a, l0, s

    vel(:) = 0.0
    y(:) = 0.0
 
    j = 0 
    do i=0, (np-1), 5
        ! Always use integer exponents, as it increases performance
        ! and has no impact on precision
        lam_ratio = (x / rest_wl(j))**2
        vel = c * (lam_ratio - 1.0) / (lam_ratio + 1.0)

        a = p(i)
        v0 = p(i+1)
        s = p(i+2)
        h3 = p(i+3)
        h4 = p(i+4)

        w = (vel - v0)/s

        alphag = exp(-w**2 / 2.0)
        hh3 = (2.0 * sq2 * w**3 - 3.0 * sq2*w) / sq6
        hh4 = (4.0 * w**4 - 12.0 * w**2 + 3.0) / sq24

        ! The observed flux density equals the emitted flux density divided by (1 + z)
        fvel = a * alphag * (1.0 + h3*hh3 + h4*hh4) / (1.0 + (vel / c))

        y = y + fvel
        j = j + 1

    enddo
    
end
