module     p0_gg_gh_d3h1l132
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity1d3h1l132.f90
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
      use p0_gg_gh_abbrevd3h1
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(6) :: acd3
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd3(1)=dotproduct(ninjaE3,spvak2k1)
      acd3(2)=dotproduct(ninjaE3,spvak2k3)
      acd3(3)=abb3(7)
      acd3(4)=abb3(9)
      acd3(5)=acd3(1)*acd3(3)
      acd3(6)=acd3(2)*acd3(4)
      acd3(5)=acd3(5)+acd3(6)
      acd3(5)=acd3(2)*acd3(5)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd3(5)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd3h1
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(18) :: acd3
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd3(1)=dotproduct(ninjaA1,spvak2k1)
      acd3(2)=dotproduct(ninjaE3,spvak2k3)
      acd3(3)=abb3(7)
      acd3(4)=dotproduct(ninjaA1,spvak2k3)
      acd3(5)=dotproduct(ninjaE3,spvak2k1)
      acd3(6)=abb3(9)
      acd3(7)=dotproduct(ninjaA0,spvak2k1)
      acd3(8)=dotproduct(ninjaA0,spvak2k3)
      acd3(9)=abb3(11)
      acd3(10)=abb3(10)
      acd3(11)=dotproduct(ninjaE3,spvak2l4)
      acd3(12)=abb3(8)
      acd3(13)=acd3(5)*acd3(3)
      acd3(14)=acd3(6)*acd3(2)
      acd3(13)=acd3(13)+2.0_ki*acd3(14)
      acd3(14)=acd3(4)*acd3(13)
      acd3(15)=acd3(3)*acd3(2)
      acd3(16)=acd3(1)*acd3(15)
      acd3(14)=acd3(16)+acd3(14)
      acd3(13)=acd3(8)*acd3(13)
      acd3(15)=acd3(7)*acd3(15)
      acd3(16)=acd3(9)*acd3(5)
      acd3(17)=acd3(10)*acd3(2)
      acd3(18)=acd3(12)*acd3(11)
      acd3(13)=acd3(18)+acd3(17)+acd3(16)+acd3(13)+acd3(15)
      brack(ninjaidxt0x0mu0)=acd3(13)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd3(14)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p0_gg_gh_d3h1_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p0_gg_gh_globalsl1, only: epspow
      use p0_gg_gh_kinematics
      use p0_gg_gh_abbrevd3h1
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA0(1:4) = + a0(0:3)
      vecA1(1:4) = + a1(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p0_gg_gh_d3h1l132
