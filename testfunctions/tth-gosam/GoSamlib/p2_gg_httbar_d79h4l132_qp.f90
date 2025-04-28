module     p2_gg_httbar_d79h4l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d79h4l132_qp.f90
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
      use p2_gg_httbar_abbrevd79h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd79
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
      use p2_gg_httbar_abbrevd79h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(46) :: acd79
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd79(1)=dotproduct(k2,ninjaE3)
      acd79(2)=dotproduct(ninjaE3,spvak2e1)
      acd79(3)=abb79(32)
      acd79(4)=dotproduct(ninjaA0,ninjaE3)
      acd79(5)=dotproduct(ninjaE3,spvae1l4)
      acd79(6)=abb79(10)
      acd79(7)=dotproduct(ninjaE3,spvae1l5)
      acd79(8)=abb79(11)
      acd79(9)=dotproduct(ninjaE3,spvae1e2)
      acd79(10)=abb79(12)
      acd79(11)=abb79(26)
      acd79(12)=dotproduct(ninjaE3,spvae2e1)
      acd79(13)=abb79(27)
      acd79(14)=dotproduct(ninjaE3,spval5e1)
      acd79(15)=abb79(39)
      acd79(16)=abb79(25)
      acd79(17)=abb79(30)
      acd79(18)=abb79(31)
      acd79(19)=dotproduct(ninjaE3,spval3l4)
      acd79(20)=dotproduct(ninjaE3,spvae1l3)
      acd79(21)=dotproduct(ninjaE3,spval3l5)
      acd79(22)=dotproduct(ninjaE3,spval3e2)
      acd79(23)=abb79(28)
      acd79(24)=abb79(15)
      acd79(25)=dotproduct(ninjaE3,spvae2k2)
      acd79(26)=abb79(33)
      acd79(27)=dotproduct(ninjaE3,spval5k2)
      acd79(28)=abb79(34)
      acd79(29)=dotproduct(ninjaE3,spvak2l4)
      acd79(30)=dotproduct(ninjaE3,spvae1k2)
      acd79(31)=abb79(18)
      acd79(32)=dotproduct(ninjaE3,spvak2e2)
      acd79(33)=abb79(23)
      acd79(34)=dotproduct(ninjaE3,spvak2l5)
      acd79(35)=abb79(35)
      acd79(36)=dotproduct(ninjaE3,spvak2l3)
      acd79(37)=dotproduct(ninjaE3,spval3e1)
      acd79(38)=dotproduct(ninjaE3,spvae2l3)
      acd79(39)=dotproduct(ninjaE3,spval5l3)
      acd79(40)=acd79(3)*acd79(1)
      acd79(41)=acd79(16)*acd79(5)
      acd79(42)=acd79(23)*acd79(7)
      acd79(43)=acd79(24)*acd79(9)
      acd79(44)=acd79(26)*acd79(25)
      acd79(45)=acd79(28)*acd79(27)
      acd79(40)=acd79(45)+acd79(44)+acd79(43)+acd79(42)+acd79(41)+acd79(40)
      acd79(40)=acd79(2)*acd79(40)
      acd79(41)=-acd79(6)*acd79(5)
      acd79(42)=acd79(8)*acd79(7)
      acd79(43)=acd79(10)*acd79(9)
      acd79(44)=acd79(11)*acd79(2)
      acd79(45)=-acd79(13)*acd79(12)
      acd79(46)=acd79(15)*acd79(14)
      acd79(41)=acd79(46)+acd79(45)+acd79(44)+acd79(43)+acd79(41)+acd79(42)
      acd79(41)=acd79(4)*acd79(41)
      acd79(42)=-acd79(19)*acd79(6)
      acd79(43)=acd79(21)*acd79(8)
      acd79(44)=acd79(22)*acd79(10)
      acd79(42)=acd79(44)+acd79(43)+acd79(42)
      acd79(42)=acd79(20)*acd79(42)
      acd79(43)=acd79(31)*acd79(29)
      acd79(44)=acd79(33)*acd79(32)
      acd79(45)=acd79(35)*acd79(34)
      acd79(43)=acd79(45)+acd79(44)+acd79(43)
      acd79(43)=acd79(30)*acd79(43)
      acd79(44)=acd79(36)*acd79(11)
      acd79(45)=-acd79(38)*acd79(13)
      acd79(46)=acd79(39)*acd79(15)
      acd79(44)=acd79(46)+acd79(45)+acd79(44)
      acd79(44)=acd79(37)*acd79(44)
      acd79(45)=acd79(17)*acd79(12)
      acd79(46)=acd79(18)*acd79(14)
      acd79(45)=acd79(46)+acd79(45)
      acd79(45)=acd79(5)*acd79(45)
      acd79(40)=2.0_ki*acd79(41)+acd79(40)+acd79(44)+acd79(43)+acd79(42)+acd79(&
      &45)
      brack(ninjaidxt0x0mu0)=acd79(40)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d79h4_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd79h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k4
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
end module     p2_gg_httbar_d79h4l132_qp
