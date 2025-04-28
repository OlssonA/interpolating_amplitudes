module     p2_gg_httbar_d264h12l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d264h12l132_qp.f90
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
      use p2_gg_httbar_abbrevd264h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd264
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
      use p2_gg_httbar_abbrevd264h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(31) :: acd264
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd264(1)=dotproduct(k2,ninjaE3)
      acd264(2)=dotproduct(ninjaE3,spvak2e2)
      acd264(3)=dotproduct(ninjaE3,spvae2e1)
      acd264(4)=abb264(58)
      acd264(5)=dotproduct(l4,ninjaE3)
      acd264(6)=dotproduct(e1,ninjaE3)
      acd264(7)=dotproduct(ninjaE3,spvae2l4)
      acd264(8)=abb264(46)
      acd264(9)=dotproduct(ninjaE3,spvae1e2)
      acd264(10)=abb264(20)
      acd264(11)=dotproduct(ninjaA0,ninjaE3)
      acd264(12)=abb264(79)
      acd264(13)=dotproduct(ninjaE3,spvae2l5)
      acd264(14)=abb264(148)
      acd264(15)=abb264(14)
      acd264(16)=abb264(115)
      acd264(17)=dotproduct(ninjaE3,spval4l5)
      acd264(18)=dotproduct(ninjaE3,spvak2l5)
      acd264(19)=dotproduct(ninjaE3,spvae2k2)
      acd264(20)=abb264(59)
      acd264(21)=dotproduct(ninjaE3,spvak2l4)
      acd264(22)=abb264(89)
      acd264(23)=abb264(149)
      acd264(24)=abb264(110)
      acd264(25)=abb264(94)
      acd264(26)=dotproduct(ninjaE3,spval4e2)
      acd264(27)=-acd264(14)*acd264(17)
      acd264(28)=acd264(5)*acd264(8)
      acd264(29)=2.0_ki*acd264(11)
      acd264(30)=acd264(12)*acd264(29)
      acd264(31)=acd264(2)*acd264(15)
      acd264(27)=acd264(31)+acd264(30)+acd264(27)+acd264(28)
      acd264(27)=acd264(7)*acd264(27)
      acd264(28)=acd264(14)*acd264(29)
      acd264(30)=acd264(2)*acd264(16)
      acd264(28)=acd264(30)+acd264(28)
      acd264(28)=acd264(13)*acd264(28)
      acd264(30)=acd264(18)*acd264(20)
      acd264(31)=acd264(21)*acd264(22)
      acd264(30)=acd264(30)+acd264(31)
      acd264(30)=acd264(19)*acd264(30)
      acd264(27)=acd264(27)+acd264(30)+acd264(28)
      acd264(27)=acd264(6)*acd264(27)
      acd264(28)=acd264(9)*acd264(19)*acd264(25)
      acd264(30)=acd264(3)*acd264(23)
      acd264(31)=acd264(26)*acd264(30)
      acd264(28)=acd264(28)+acd264(31)
      acd264(28)=acd264(21)*acd264(28)
      acd264(30)=-acd264(29)*acd264(30)
      acd264(31)=acd264(3)*acd264(1)*acd264(4)
      acd264(30)=acd264(31)+acd264(30)
      acd264(30)=acd264(2)*acd264(30)
      acd264(31)=-acd264(5)*acd264(10)
      acd264(29)=acd264(24)*acd264(29)
      acd264(29)=acd264(31)+acd264(29)
      acd264(29)=acd264(7)*acd264(9)*acd264(29)
      acd264(27)=acd264(27)+acd264(29)+acd264(30)+acd264(28)
      brack(ninjaidxt0x0mu0)=acd264(27)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d264h12_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd264h12_qp
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
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d264h12l132_qp
