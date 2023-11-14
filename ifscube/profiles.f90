subroutine gauss(x, p, y, n, np)

    implicit none
    integer, parameter :: dp = selected_real_kind(15, 307)
    integer :: n
    integer :: np
    integer :: i
    !f2py intent(hide), depend(x) :: n=len(x)
    !f2py integer intent(hide), depend(p) :: np=len(p)
    real(dp), parameter :: pi = 3.1415927
    real(dp), dimension(n), intent(in) :: x
    real(dp), dimension(np), intent(in) :: p
    real(dp), dimension(n), intent(out) :: y
    real(dp), dimension(n) :: w
    real(dp), dimension(n) :: g

    y(:) = 0.0

    do i = 1, np, 3
        w = -((x - p(i + 1)) / p(i + 2))**2
        g = p(i) * exp(w / 2.0)
        y = y + g
    enddo

end


subroutine gauss_hermite(x, p, y, n, np)

    implicit none
    integer, parameter :: dp = selected_real_kind(15, 307)
    !f2py intent(hide), depend(x) :: n=len(x)
    !f2py integer intent(hide), depend(p) :: np=len(p)
    integer :: n
    integer :: np
    integer :: i
    real(dp), parameter :: pi = 3.1415927
    real(dp), parameter :: sq2 = sqrt(2.0)
    real(dp), parameter :: sq6 = sqrt(6.0)
    real(dp), parameter :: sq24 = sqrt(24.0)
    real(dp), parameter :: sq2pi = sqrt(2.0 * pi)
    real(dp), dimension(np), intent(in) :: p
    real(dp), dimension(n), intent(in) :: x
    real(dp), dimension(n), intent(out) :: y
    real(dp), dimension(n) :: w
    real(dp), dimension(n) :: h3
    real(dp), dimension(n) :: hh3
    real(dp), dimension(n) :: h4
    real(dp), dimension(n) :: hh4
    real(dp), dimension(n) :: alphag
    real(dp), dimension(n) :: gh
    real(dp) :: a
    real(dp) :: l0
    real(dp) :: s

    y(:) = 0.0

    do i = 1, np, 5

        a = p(i)
        l0 = p(i + 1)
        s = p(i + 2)
        h3 = p(i + 3)
        h4 = p(i + 4)

        w = (x - l0) / s

        alphag = 1. / sq2pi * exp(-w**2 / 2.)
        hh3 = 1. / sq6 * (2.0 * sq2 * w**3 - 3.0 * sq2 * w)
        hh4 = 1. / sq24 * (4.0 * w**4 - 12.0 * w**2 + 3.0)

        gh = a * alphag / s * (1.0 + h3 * hh3 + h4 * hh4)

        y = y + gh

    enddo

end

subroutine gauss_vel(x, rest_wl, p, y, n, np, nwl)

    implicit none
    integer, parameter :: dp = selected_real_kind(15, 307)
    integer :: n
    integer :: np
    integer :: nwl
    integer :: j
    integer :: i
    !f2py intent(hide), depend(x) :: n=len(x)
    !f2py integer intent(hide), depend(p) :: np=len(p)
    !f2py integer intent(hide), depend(rest_wl) :: np=len(rest_wl)
    real(dp), parameter :: pi = 3.1415927
    real(dp), parameter :: c = 299792.458
    real(dp), dimension(np), intent(in) :: p
    real(dp), dimension(nwl), intent(in) :: rest_wl
    real(dp), dimension(n), intent(in) :: x
    real(dp), dimension(n), intent(out) :: y
    real(dp), dimension(n) :: w
    real(dp), dimension(n) :: vel
    real(dp), dimension(n) :: fvel
    real(dp), dimension(n) :: lam_ratio

    vel(:) = 0.0
    y(:) = 0.0

    j = 1
    do i = 1, np, 3
        lam_ratio = (x / rest_wl(j)) ** 2
        vel = c * (lam_ratio - 1.0) / (lam_ratio + 1.0)

        w = -((vel - p(i + 1)) / p(i + 2))**2 / 2.0
        ! The observed flux density equals the emitted flux density divided by (1 + z)
        fvel = p(i) * exp(w) / (1.0 + (vel / c))

        y = y + fvel
        j = j + 1

    enddo

end

subroutine gauss_hermite_vel(x, rest_wl, p, y, n, np, nwl)

    implicit none
    integer, parameter :: dp = selected_real_kind(15, 307)
    integer :: n=len(x)
    integer :: np
    integer :: nwl
    integer :: j
    integer :: i
    !f2py intent(hide), depend(x) :: n=len(x)
    !f2py integer intent(hide), depend(p) :: np=len(p)
    !f2py integer intent(hide), depend(rest_wl) :: np=len(rest_wl)
    real(dp), parameter :: pi = 3.1415927
    real(dp), parameter :: sq2 = sqrt(2.0)
    real(dp), parameter :: sq6 = sqrt(6.0)
    real(dp), parameter :: sq24 = sqrt(24.0)
    real(dp), parameter :: sq2pi = sqrt(2.0 * pi)
    real(dp), parameter :: c = 299792.458
    real(dp), dimension(nwl), intent(in) :: rest_wl
    real(dp), dimension(np), intent(in) :: p
    real(dp), dimension(n), intent(in) :: x
    real(dp), dimension(n), intent(out) :: y
    real(dp), dimension(n) :: w
    real(dp), dimension(n) :: hh3
    real(dp), dimension(n) :: hh4
    real(dp), dimension(n) :: alphag
    real(dp), dimension(n) :: vel
    real(dp), dimension(n) :: fvel
    real(dp), dimension(n) :: lam_ratio
    real(dp) :: a
    real(dp) :: v0
    real(dp) :: s
    real(dp) :: h3
    real(dp) :: h4

    vel(:) = 0.0
    y(:) = 0.0

    j = 1
    do i = 1, np, 5
        ! Always use integer exponents, as it increases performance
        ! and has no impact on precision
        lam_ratio = (x / rest_wl(j))**2
        vel = c * (lam_ratio - 1.0) / (lam_ratio + 1.0)

        a = p(i)
        v0 = p(i + 1)
        s = p(i + 2)
        h3 = p(i + 3)
        h4 = p(i + 4)

        w = (vel - v0) / s

        alphag = exp(-w**2 / 2.0)
        hh3 = (2.0 * sq2 * w**3 - 3.0 * sq2 * w) / sq6
        hh4 = (4.0 * w**4 - 12.0 * w**2 + 3.0) / sq24

        ! The observed flux density equals the emitted flux density divided by (1 + z)
        fvel = a * alphag * (1.0 + h3 * hh3 + h4 * hh4) / (1.0 + (vel / c))

        y = y + fvel
        j = j + 1

    enddo

end
