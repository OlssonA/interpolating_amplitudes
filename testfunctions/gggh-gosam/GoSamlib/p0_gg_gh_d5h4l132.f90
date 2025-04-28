module     p0_gg_gh_d5h4l132
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity4d5h4l132.f90
   ! generator: buildfortran_tn3.py
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2x0mu0 = 0
   integer, parameter :: ninjaidxt1x0mu0 = 1
   integer, parameter :: ninjaidxt1x1mu0 = 2
   integer, parameter :: ninjaidxt0x0mu0 = 3
   integer, parameter :: ninjaidxt0x0mu2 = 4
   integer, parameter :: ninjaidxt0x1mu0 = 5
   integer, parameter :: ninjaidxt0x2mu0 = 6
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd5h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(8) :: acd5
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd5(1)=dotproduct(k2,ninjaE3)
      acd5(2)=dotproduct(ninjaE3,spvak1k2)
      acd5(3)=abb5(9)
      acd5(4)=dotproduct(ninjaE3,spvak1k3)
      acd5(5)=dotproduct(ninjaE3,spvak3k2)
      acd5(6)=abb5(11)
      acd5(7)=acd5(3)*acd5(2)*acd5(1)
      acd5(8)=acd5(6)*acd5(5)*acd5(4)
      acd5(7)=acd5(7)+acd5(8)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd5(7)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd5h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(36) :: acd5
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd5(1)=dotproduct(k2,ninjaA1)
      acd5(2)=dotproduct(ninjaE3,spvak1k2)
      acd5(3)=abb5(9)
      acd5(4)=dotproduct(k2,ninjaE3)
      acd5(5)=dotproduct(ninjaA1,spvak1k2)
      acd5(6)=dotproduct(ninjaA1,spvak1k3)
      acd5(7)=dotproduct(ninjaE3,spvak3k2)
      acd5(8)=abb5(11)
      acd5(9)=dotproduct(ninjaA1,spvak3k2)
      acd5(10)=dotproduct(ninjaE3,spvak1k3)
      acd5(11)=dotproduct(k2,ninjaA0)
      acd5(12)=dotproduct(ninjaA0,spvak1k2)
      acd5(13)=abb5(15)
      acd5(14)=dotproduct(ninjaA0,ninjaE3)
      acd5(15)=abb5(16)
      acd5(16)=dotproduct(ninjaA0,spvak1k3)
      acd5(17)=dotproduct(ninjaA0,spvak3k2)
      acd5(18)=abb5(7)
      acd5(19)=dotproduct(ninjaE3,spval4k2)
      acd5(20)=abb5(8)
      acd5(21)=abb5(17)
      acd5(22)=dotproduct(ninjaE3,spvak1l4)
      acd5(23)=abb5(10)
      acd5(24)=abb5(14)
      acd5(25)=acd5(2)*acd5(3)
      acd5(26)=acd5(1)*acd5(25)
      acd5(27)=acd5(4)*acd5(3)
      acd5(28)=acd5(5)*acd5(27)
      acd5(29)=acd5(7)*acd5(8)
      acd5(30)=acd5(6)*acd5(29)
      acd5(31)=acd5(10)*acd5(8)
      acd5(32)=acd5(9)*acd5(31)
      acd5(26)=acd5(32)+acd5(30)+acd5(26)+acd5(28)
      acd5(25)=acd5(11)*acd5(25)
      acd5(27)=acd5(12)*acd5(27)
      acd5(28)=acd5(13)*acd5(4)
      acd5(30)=acd5(15)*acd5(14)
      acd5(29)=acd5(16)*acd5(29)
      acd5(31)=acd5(17)*acd5(31)
      acd5(32)=acd5(18)*acd5(10)
      acd5(33)=acd5(20)*acd5(19)
      acd5(34)=acd5(21)*acd5(2)
      acd5(35)=acd5(23)*acd5(22)
      acd5(36)=acd5(24)*acd5(7)
      acd5(25)=acd5(36)+acd5(35)+acd5(34)+acd5(33)+acd5(32)+acd5(31)+acd5(29)+2&
      &.0_ki*acd5(30)+acd5(28)+acd5(25)+acd5(27)
      brack(ninjaidxt0x0mu0)=acd5(25)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd5(26)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p0_gg_gh_d5h4_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p0_gg_gh_globalsl1, only: epspow
      use p0_gg_gh_kinematics
      use p0_gg_gh_abbrevd5h4
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k4-k3
      vecA0(1:4) = - a0(0:3) - qshift(1:4)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p0_gg_gh_d5h4l132
