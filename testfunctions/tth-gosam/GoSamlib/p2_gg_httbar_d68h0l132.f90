module     p2_gg_httbar_d68h0l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d68h0l132.f90
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
      use p2_gg_httbar_abbrevd68h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd68
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
      use p2_gg_httbar_abbrevd68h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(19) :: acd68
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd68(1)=dotproduct(k2,ninjaE3)
      acd68(2)=dotproduct(ninjaE3,spval4k2)
      acd68(3)=abb68(25)
      acd68(4)=dotproduct(ninjaE3,spval5k2)
      acd68(5)=abb68(48)
      acd68(6)=dotproduct(ninjaA0,ninjaE3)
      acd68(7)=dotproduct(ninjaE3,spval4k1)
      acd68(8)=abb68(26)
      acd68(9)=abb68(28)
      acd68(10)=dotproduct(ninjaE3,spval5k1)
      acd68(11)=abb68(38)
      acd68(12)=abb68(45)
      acd68(13)=dotproduct(ninjaE3,spvak1k2)
      acd68(14)=abb68(22)
      acd68(15)=abb68(47)
      acd68(16)=acd68(8)*acd68(7)
      acd68(17)=acd68(9)*acd68(2)
      acd68(18)=acd68(11)*acd68(10)
      acd68(19)=acd68(12)*acd68(4)
      acd68(16)=acd68(19)+acd68(18)+acd68(17)+acd68(16)
      acd68(17)=2.0_ki*acd68(6)
      acd68(16)=acd68(17)*acd68(16)
      acd68(17)=acd68(3)*acd68(2)
      acd68(18)=acd68(5)*acd68(4)
      acd68(17)=acd68(17)+acd68(18)
      acd68(17)=acd68(1)*acd68(17)
      acd68(18)=acd68(14)*acd68(7)
      acd68(19)=-acd68(15)*acd68(10)
      acd68(18)=acd68(19)+acd68(18)
      acd68(18)=acd68(13)*acd68(18)
      acd68(16)=acd68(16)+acd68(18)+acd68(17)
      brack(ninjaidxt0x0mu0)=acd68(16)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d68h0_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd68h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k5
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
end module     p2_gg_httbar_d68h0l132
