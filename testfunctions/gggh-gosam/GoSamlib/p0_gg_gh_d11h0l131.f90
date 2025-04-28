module     p0_gg_gh_d11h0l131
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity0d11h0l131.f90
   ! generator: buildfortran_tn3.py
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt3mu0 = 0
   integer, parameter :: ninjaidxt2mu0 = 1
   integer, parameter :: ninjaidxt1mu0 = 2
   integer, parameter :: ninjaidxt1mu2 = 3
   integer, parameter :: ninjaidxt0mu0 = 4
   integer, parameter :: ninjaidxt0mu2 = 5
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd11h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(4) :: acd11
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd11(1)=dotproduct(ninjaE3,spvak1k2)
      acd11(2)=dotproduct(ninjaE3,spvak2k3)
      acd11(3)=abb11(10)
      acd11(4)=acd11(3)*acd11(1)*acd11(2)**2
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd11(4)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd11h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(37) :: acd11
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd11(1)=dotproduct(ninjaE3,spvak2k3)
      acd11(2)=dotproduct(ninjaE4,spvak1k2)
      acd11(3)=abb11(10)
      acd11(4)=dotproduct(ninjaE3,spvak1k2)
      acd11(5)=dotproduct(ninjaE4,spvak2k3)
      acd11(6)=abb11(12)
      acd11(7)=dotproduct(k1,ninjaE3)
      acd11(8)=dotproduct(k2,ninjaE3)
      acd11(9)=abb11(11)
      acd11(10)=dotproduct(ninjaA,ninjaE3)
      acd11(11)=dotproduct(ninjaA,spvak2k3)
      acd11(12)=dotproduct(ninjaA,spvak1k2)
      acd11(13)=dotproduct(ninjaE3,spvak1l4)
      acd11(14)=abb11(8)
      acd11(15)=abb11(13)
      acd11(16)=dotproduct(ninjaE3,spval4k2)
      acd11(17)=abb11(14)
      acd11(18)=dotproduct(ninjaE3,spvak1k3)
      acd11(19)=abb11(15)
      acd11(20)=dotproduct(k1,ninjaA)
      acd11(21)=dotproduct(k2,ninjaA)
      acd11(22)=dotproduct(ninjaA,ninjaA)
      acd11(23)=dotproduct(ninjaA,spvak1l4)
      acd11(24)=dotproduct(ninjaA,spval4k2)
      acd11(25)=dotproduct(ninjaA,spvak1k3)
      acd11(26)=abb11(9)
      acd11(27)=acd11(3)*acd11(4)
      acd11(28)=2.0_ki*acd11(27)
      acd11(29)=acd11(28)*acd11(5)
      acd11(30)=acd11(1)*acd11(3)
      acd11(31)=acd11(30)*acd11(2)
      acd11(29)=acd11(29)+acd11(31)
      acd11(31)=acd11(6)+acd11(29)
      acd11(31)=acd11(1)*acd11(31)
      acd11(32)=acd11(7)+2.0_ki*acd11(10)
      acd11(32)=acd11(32)*acd11(6)
      acd11(33)=acd11(19)*acd11(18)
      acd11(34)=acd11(17)*acd11(16)
      acd11(35)=acd11(14)*acd11(13)
      acd11(36)=acd11(9)*acd11(8)
      acd11(37)=acd11(4)*acd11(15)
      acd11(32)=acd11(32)+acd11(37)+acd11(33)+acd11(34)+acd11(35)+acd11(36)
      acd11(28)=acd11(11)*acd11(28)
      acd11(30)=acd11(12)*acd11(30)
      acd11(28)=acd11(30)+acd11(28)+acd11(32)
      acd11(28)=acd11(1)*acd11(28)
      acd11(29)=ninjaP*acd11(29)
      acd11(30)=2.0_ki*acd11(3)
      acd11(30)=acd11(11)*acd11(30)
      acd11(30)=acd11(30)+acd11(15)
      acd11(30)=acd11(12)*acd11(30)
      acd11(33)=acd11(19)*acd11(25)
      acd11(34)=acd11(17)*acd11(24)
      acd11(35)=acd11(14)*acd11(23)
      acd11(36)=acd11(9)*acd11(21)
      acd11(37)=ninjaP+acd11(22)+acd11(20)
      acd11(37)=acd11(6)*acd11(37)
      acd11(29)=acd11(37)+acd11(36)+acd11(35)+acd11(34)+acd11(26)+acd11(33)+acd&
      &11(30)+acd11(29)
      acd11(29)=acd11(1)*acd11(29)
      acd11(27)=acd11(27)*acd11(11)
      acd11(27)=acd11(27)+acd11(32)
      acd11(27)=acd11(11)*acd11(27)
      acd11(27)=acd11(27)+acd11(29)
      brack(ninjaidxt1mu0)=acd11(28)
      brack(ninjaidxt1mu2)=0.0_ki
      brack(ninjaidxt0mu0)=acd11(27)
      brack(ninjaidxt0mu2)=acd11(31)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_gg_gh_d11h0_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p0_gg_gh_globalsl1, only: epspow
      use p0_gg_gh_kinematics
      use p0_gg_gh_abbrevd11h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_gg_gh_d11h0l131
