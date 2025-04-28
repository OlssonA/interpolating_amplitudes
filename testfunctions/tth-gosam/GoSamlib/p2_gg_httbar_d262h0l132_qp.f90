module     p2_gg_httbar_d262h0l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d262h0l132_qp.f90
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
      use p2_gg_httbar_abbrevd262h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd262
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
      use p2_gg_httbar_abbrevd262h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(32) :: acd262
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd262(1)=dotproduct(k2,ninjaE3)
      acd262(2)=dotproduct(ninjaE3,spvae1k2)
      acd262(3)=dotproduct(ninjaE3,spvae2e1)
      acd262(4)=abb262(18)
      acd262(5)=dotproduct(e2,ninjaE3)
      acd262(6)=dotproduct(ninjaA0,ninjaE3)
      acd262(7)=dotproduct(ninjaE3,spval5e1)
      acd262(8)=abb262(94)
      acd262(9)=dotproduct(ninjaE3,spval4e1)
      acd262(10)=abb262(115)
      acd262(11)=abb262(33)
      acd262(12)=abb262(35)
      acd262(13)=dotproduct(ninjaE3,spval4k2)
      acd262(14)=dotproduct(ninjaE3,spvak2e1)
      acd262(15)=abb262(24)
      acd262(16)=dotproduct(ninjaE3,spval5k2)
      acd262(17)=abb262(71)
      acd262(18)=dotproduct(ninjaE3,spval5l3)
      acd262(19)=dotproduct(ninjaE3,spval3e1)
      acd262(20)=dotproduct(ninjaE3,spval4l3)
      acd262(21)=abb262(26)
      acd262(22)=dotproduct(ninjaE3,spvae1e2)
      acd262(23)=abb262(118)
      acd262(24)=dotproduct(ninjaE3,spval3k2)
      acd262(25)=dotproduct(ninjaE3,spvae1l3)
      acd262(26)=abb262(56)
      acd262(27)=-acd262(10)*acd262(20)
      acd262(28)=acd262(8)*acd262(18)
      acd262(27)=acd262(27)+acd262(28)
      acd262(27)=acd262(19)*acd262(27)
      acd262(28)=acd262(16)*acd262(17)
      acd262(29)=acd262(13)*acd262(15)
      acd262(28)=acd262(28)+acd262(29)
      acd262(28)=acd262(14)*acd262(28)
      acd262(29)=-acd262(7)*acd262(8)
      acd262(30)=acd262(9)*acd262(10)
      acd262(29)=acd262(29)+acd262(30)
      acd262(30)=2.0_ki*acd262(6)
      acd262(29)=acd262(29)*acd262(30)
      acd262(31)=acd262(7)*acd262(11)
      acd262(32)=acd262(9)*acd262(12)
      acd262(31)=acd262(31)+acd262(32)
      acd262(31)=acd262(2)*acd262(31)
      acd262(27)=acd262(31)+acd262(29)+acd262(27)+acd262(28)
      acd262(27)=acd262(5)*acd262(27)
      acd262(28)=acd262(19)*acd262(20)
      acd262(29)=-acd262(9)*acd262(30)
      acd262(28)=acd262(29)+acd262(28)
      acd262(28)=acd262(28)*acd262(23)
      acd262(29)=acd262(14)*acd262(13)*acd262(26)
      acd262(28)=acd262(29)+acd262(28)
      acd262(28)=acd262(22)*acd262(28)
      acd262(29)=acd262(3)*acd262(21)
      acd262(30)=acd262(30)*acd262(29)
      acd262(31)=acd262(3)*acd262(1)*acd262(4)
      acd262(30)=acd262(31)+acd262(30)
      acd262(30)=acd262(2)*acd262(30)
      acd262(29)=-acd262(24)*acd262(25)*acd262(29)
      acd262(27)=acd262(27)+acd262(30)+acd262(29)+acd262(28)
      brack(ninjaidxt0x0mu0)=acd262(27)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d262h0_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd262h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3-k4
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
end module     p2_gg_httbar_d262h0l132_qp
