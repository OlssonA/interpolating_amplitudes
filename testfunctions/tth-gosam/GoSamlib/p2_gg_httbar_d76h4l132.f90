module     p2_gg_httbar_d76h4l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d76h4l132.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt1x0mu0 = 0
   integer, parameter :: ninjaidxt0x0mu0 = 1
   integer, parameter :: ninjaidxt0x1mu0 = 2
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd76h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd76
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      brack(ninjaidxt1x0mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd76h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(34) :: acd76
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd76(1)=dotproduct(ninjaA0,ninjaE3)
      acd76(2)=dotproduct(ninjaE3,spvae1k2)
      acd76(3)=abb76(26)
      acd76(4)=dotproduct(ninjaE3,spval4e1)
      acd76(5)=abb76(20)
      acd76(6)=dotproduct(ninjaE3,spval5e1)
      acd76(7)=abb76(19)
      acd76(8)=dotproduct(ninjaE3,spvae2e1)
      acd76(9)=abb76(13)
      acd76(10)=dotproduct(ninjaE3,spvae1e2)
      acd76(11)=abb76(22)
      acd76(12)=dotproduct(ninjaE3,spvae1l4)
      acd76(13)=abb76(25)
      acd76(14)=abb76(10)
      acd76(15)=abb76(12)
      acd76(16)=abb76(15)
      acd76(17)=dotproduct(ninjaE3,spval3e1)
      acd76(18)=abb76(27)
      acd76(19)=dotproduct(ninjaE3,spvak2e1)
      acd76(20)=abb76(29)
      acd76(21)=dotproduct(ninjaE3,spvae1l3)
      acd76(22)=abb76(40)
      acd76(23)=abb76(24)
      acd76(24)=abb76(37)
      acd76(25)=abb76(38)
      acd76(26)=abb76(39)
      acd76(27)=abb76(21)
      acd76(28)=abb76(41)
      acd76(29)=acd76(3)*acd76(2)
      acd76(30)=-acd76(5)*acd76(4)
      acd76(31)=acd76(7)*acd76(6)
      acd76(32)=-acd76(9)*acd76(8)
      acd76(33)=acd76(11)*acd76(10)
      acd76(34)=-acd76(13)*acd76(12)
      acd76(29)=acd76(34)+acd76(33)+acd76(32)+acd76(31)+acd76(29)+acd76(30)
      acd76(29)=acd76(1)*acd76(29)
      acd76(30)=acd76(14)*acd76(4)
      acd76(31)=acd76(15)*acd76(6)
      acd76(32)=acd76(16)*acd76(8)
      acd76(33)=acd76(18)*acd76(17)
      acd76(34)=acd76(20)*acd76(19)
      acd76(30)=acd76(34)+acd76(33)+acd76(32)+acd76(31)+acd76(30)
      acd76(30)=acd76(2)*acd76(30)
      acd76(31)=acd76(22)*acd76(4)
      acd76(32)=-acd76(25)*acd76(6)
      acd76(33)=acd76(26)*acd76(8)
      acd76(31)=acd76(33)+acd76(32)+acd76(31)
      acd76(31)=acd76(21)*acd76(31)
      acd76(32)=acd76(23)*acd76(10)
      acd76(33)=acd76(24)*acd76(12)
      acd76(32)=acd76(33)+acd76(32)
      acd76(32)=acd76(6)*acd76(32)
      acd76(33)=-acd76(27)*acd76(10)
      acd76(34)=acd76(28)*acd76(12)
      acd76(33)=acd76(34)+acd76(33)
      acd76(33)=acd76(17)*acd76(33)
      acd76(29)=2.0_ki*acd76(29)+acd76(30)+acd76(31)+acd76(33)+acd76(32)
      brack(ninjaidxt0x0mu0)=acd76(29)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d76h4_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd76h4
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k5
      vecA0(1:4) = - a0(0:3) - qshift(1:4)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d76h4l132
