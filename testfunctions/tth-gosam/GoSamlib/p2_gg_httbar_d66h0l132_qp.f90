module     p2_gg_httbar_d66h0l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d66h0l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd66h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd66
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd66h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(33) :: acd66
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd66(1)=dotproduct(k1,ninjaE3)
      acd66(2)=dotproduct(ninjaA0,ninjaE3)
      acd66(3)=abb66(38)
      acd66(4)=dotproduct(ninjaE3,spval4k2)
      acd66(5)=abb66(48)
      acd66(6)=dotproduct(ninjaE3,spval5k2)
      acd66(7)=abb66(63)
      acd66(8)=dotproduct(ninjaE3,spval5l3)
      acd66(9)=abb66(22)
      acd66(10)=dotproduct(ninjaE3,spval4l3)
      acd66(11)=abb66(71)
      acd66(12)=dotproduct(k2,ninjaE3)
      acd66(13)=dotproduct(ninjaE3,spvak1k2)
      acd66(14)=abb66(14)
      acd66(15)=abb66(19)
      acd66(16)=abb66(12)
      acd66(17)=abb66(21)
      acd66(18)=dotproduct(ninjaE3,spval5k1)
      acd66(19)=abb66(28)
      acd66(20)=dotproduct(ninjaE3,spval4k1)
      acd66(21)=abb66(30)
      acd66(22)=dotproduct(ninjaE3,spvak2k1)
      acd66(23)=abb66(31)
      acd66(24)=dotproduct(ninjaE3,spval3k2)
      acd66(25)=dotproduct(ninjaE3,spvak1l3)
      acd66(26)=abb66(25)
      acd66(27)=dotproduct(ninjaE3,spval3k1)
      acd66(28)=-acd66(3)*acd66(2)
      acd66(29)=acd66(5)*acd66(4)
      acd66(30)=acd66(9)*acd66(8)
      acd66(31)=-acd66(11)*acd66(10)
      acd66(28)=2.0_ki*acd66(28)+acd66(31)+acd66(30)+acd66(29)
      acd66(29)=acd66(1)-acd66(12)
      acd66(28)=acd66(29)*acd66(28)
      acd66(29)=acd66(17)*acd66(6)
      acd66(30)=acd66(16)*acd66(4)
      acd66(31)=-acd66(18)*acd66(19)
      acd66(32)=acd66(21)*acd66(20)
      acd66(29)=acd66(32)+acd66(31)+acd66(30)+acd66(29)
      acd66(30)=2.0_ki*acd66(2)
      acd66(29)=acd66(30)*acd66(29)
      acd66(30)=acd66(7)*acd66(1)
      acd66(31)=acd66(15)*acd66(12)
      acd66(32)=-acd66(23)*acd66(22)
      acd66(30)=acd66(32)+acd66(31)+acd66(30)
      acd66(30)=acd66(6)*acd66(30)
      acd66(31)=acd66(17)*acd66(8)
      acd66(32)=acd66(26)*acd66(25)
      acd66(31)=acd66(32)+acd66(31)
      acd66(31)=acd66(24)*acd66(31)
      acd66(32)=acd66(14)*acd66(13)*acd66(12)
      acd66(33)=-acd66(27)*acd66(19)*acd66(8)
      acd66(28)=acd66(33)+acd66(32)+acd66(29)+acd66(30)+acd66(31)+acd66(28)
      brack(ninjaidxt0x0mu0)=acd66(28)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d66h0_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd66h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k5
      vecA0(1:4) = + a0(0:3) - qshift(1:4)
      vecA1(1:4) = + a1(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d66h0l132_qp
