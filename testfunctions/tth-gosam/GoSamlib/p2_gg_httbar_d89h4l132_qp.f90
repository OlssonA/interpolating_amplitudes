module     p2_gg_httbar_d89h4l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d89h4l132_qp.f90
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
      use p2_gg_httbar_abbrevd89h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(15) :: acd89
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd89(1)=dotproduct(ninjaE3,spvae1l4)
      acd89(2)=dotproduct(ninjaE3,spval5e2)
      acd89(3)=dotproduct(ninjaE3,spvae2e1)
      acd89(4)=abb89(11)
      acd89(5)=dotproduct(ninjaE3,spval3e2)
      acd89(6)=abb89(19)
      acd89(7)=dotproduct(ninjaE3,spvak2e1)
      acd89(8)=dotproduct(ninjaE3,spvae2k2)
      acd89(9)=dotproduct(ninjaE3,spvae1e2)
      acd89(10)=abb89(17)
      acd89(11)=dotproduct(ninjaE3,spvae2l3)
      acd89(12)=abb89(49)
      acd89(13)=-acd89(10)*acd89(8)
      acd89(14)=-acd89(12)*acd89(11)
      acd89(13)=acd89(14)+acd89(13)
      acd89(13)=acd89(13)*acd89(9)*acd89(7)
      acd89(14)=acd89(4)*acd89(2)
      acd89(15)=acd89(6)*acd89(5)
      acd89(14)=acd89(14)+acd89(15)
      acd89(14)=acd89(14)*acd89(3)*acd89(1)
      acd89(13)=acd89(14)+acd89(13)
      brack(ninjaidxt1x0mu0)=acd89(13)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd89h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(47) :: acd89
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd89(1)=dotproduct(ninjaA1,spval5e2)
      acd89(2)=dotproduct(ninjaE3,spvae2e1)
      acd89(3)=dotproduct(ninjaE3,spvae1l4)
      acd89(4)=abb89(11)
      acd89(5)=dotproduct(ninjaA1,spvae2e1)
      acd89(6)=dotproduct(ninjaE3,spval5e2)
      acd89(7)=dotproduct(ninjaE3,spval3e2)
      acd89(8)=abb89(19)
      acd89(9)=dotproduct(ninjaA1,spvae1l4)
      acd89(10)=dotproduct(ninjaA1,spvak2e1)
      acd89(11)=dotproduct(ninjaE3,spvae2k2)
      acd89(12)=dotproduct(ninjaE3,spvae1e2)
      acd89(13)=abb89(17)
      acd89(14)=dotproduct(ninjaE3,spvae2l3)
      acd89(15)=abb89(49)
      acd89(16)=dotproduct(ninjaA1,spvae2k2)
      acd89(17)=dotproduct(ninjaE3,spvak2e1)
      acd89(18)=dotproduct(ninjaA1,spvae1e2)
      acd89(19)=dotproduct(ninjaA1,spval3e2)
      acd89(20)=dotproduct(ninjaA1,spvae2l3)
      acd89(21)=dotproduct(ninjaA0,spval5e2)
      acd89(22)=dotproduct(ninjaA0,spvae2e1)
      acd89(23)=dotproduct(ninjaA0,spvae1l4)
      acd89(24)=dotproduct(ninjaA0,spvak2e1)
      acd89(25)=dotproduct(ninjaA0,spvae2k2)
      acd89(26)=dotproduct(ninjaA0,spvae1e2)
      acd89(27)=dotproduct(ninjaA0,spval3e2)
      acd89(28)=dotproduct(ninjaA0,spvae2l3)
      acd89(29)=abb89(10)
      acd89(30)=abb89(12)
      acd89(31)=abb89(50)
      acd89(32)=dotproduct(ninjaE3,spvak2e2)
      acd89(33)=abb89(31)
      acd89(34)=abb89(41)
      acd89(35)=abb89(32)
      acd89(36)=abb89(26)
      acd89(37)=dotproduct(ninjaE3,spvae2l4)
      acd89(38)=abb89(43)
      acd89(39)=acd89(15)*acd89(14)
      acd89(40)=acd89(13)*acd89(11)
      acd89(39)=acd89(39)+acd89(40)
      acd89(40)=-acd89(10)*acd89(39)
      acd89(41)=-acd89(15)*acd89(20)
      acd89(42)=-acd89(13)*acd89(16)
      acd89(41)=acd89(41)+acd89(42)
      acd89(41)=acd89(17)*acd89(41)
      acd89(40)=acd89(41)+acd89(40)
      acd89(40)=acd89(12)*acd89(40)
      acd89(41)=acd89(8)*acd89(7)
      acd89(42)=acd89(4)*acd89(6)
      acd89(41)=acd89(41)+acd89(42)
      acd89(42)=acd89(9)*acd89(41)
      acd89(43)=acd89(8)*acd89(19)
      acd89(44)=acd89(4)*acd89(1)
      acd89(43)=acd89(43)+acd89(44)
      acd89(43)=acd89(3)*acd89(43)
      acd89(42)=acd89(43)+acd89(42)
      acd89(42)=acd89(2)*acd89(42)
      acd89(43)=acd89(39)*acd89(17)
      acd89(44)=-acd89(18)*acd89(43)
      acd89(45)=acd89(41)*acd89(3)
      acd89(46)=acd89(5)*acd89(45)
      acd89(40)=acd89(42)+acd89(40)+acd89(44)+acd89(46)
      acd89(39)=-acd89(24)*acd89(39)
      acd89(42)=-acd89(15)*acd89(28)
      acd89(44)=-acd89(13)*acd89(25)
      acd89(42)=acd89(44)+acd89(34)+acd89(42)
      acd89(42)=acd89(17)*acd89(42)
      acd89(44)=acd89(37)*acd89(38)
      acd89(46)=acd89(14)*acd89(36)
      acd89(47)=acd89(11)*acd89(35)
      acd89(39)=acd89(42)+acd89(47)+acd89(44)+acd89(46)+acd89(39)
      acd89(39)=acd89(12)*acd89(39)
      acd89(41)=acd89(23)*acd89(41)
      acd89(42)=acd89(8)*acd89(27)
      acd89(44)=acd89(4)*acd89(21)
      acd89(42)=acd89(44)+acd89(30)+acd89(42)
      acd89(42)=acd89(3)*acd89(42)
      acd89(44)=acd89(32)*acd89(33)
      acd89(46)=acd89(7)*acd89(31)
      acd89(47)=acd89(6)*acd89(29)
      acd89(41)=acd89(42)+acd89(47)+acd89(44)+acd89(46)+acd89(41)
      acd89(41)=acd89(2)*acd89(41)
      acd89(42)=-acd89(26)*acd89(43)
      acd89(43)=acd89(22)*acd89(45)
      acd89(39)=acd89(41)+acd89(39)+acd89(42)+acd89(43)
      brack(ninjaidxt0x0mu0)=acd89(39)
      brack(ninjaidxt0x1mu0)=acd89(40)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d89h4_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd89h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k2+k4
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
end module     p2_gg_httbar_d89h4l132_qp
