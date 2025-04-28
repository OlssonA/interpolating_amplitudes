module     p2_gg_httbar_d162h4l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d162h4l132_qp.f90
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
      use p2_gg_httbar_abbrevd162h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(24) :: acd162
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd162(1)=dotproduct(ninjaE3,spvak1e1)
      acd162(2)=dotproduct(ninjaE3,spvae1k2)
      acd162(3)=abb162(13)
      acd162(4)=dotproduct(ninjaE3,spval5e1)
      acd162(5)=abb162(37)
      acd162(6)=dotproduct(ninjaE3,spvae2e1)
      acd162(7)=abb162(23)
      acd162(8)=dotproduct(ninjaE3,spvak2e1)
      acd162(9)=abb162(26)
      acd162(10)=dotproduct(ninjaE3,spval4e1)
      acd162(11)=abb162(43)
      acd162(12)=dotproduct(ninjaE3,spvae1e2)
      acd162(13)=abb162(17)
      acd162(14)=dotproduct(ninjaE3,spvae1l5)
      acd162(15)=abb162(27)
      acd162(16)=dotproduct(ninjaE3,spvae1k1)
      acd162(17)=abb162(30)
      acd162(18)=dotproduct(ninjaE3,spvae1l4)
      acd162(19)=abb162(39)
      acd162(20)=acd162(5)*acd162(2)
      acd162(21)=acd162(13)*acd162(12)
      acd162(22)=acd162(15)*acd162(14)
      acd162(23)=acd162(17)*acd162(16)
      acd162(24)=acd162(19)*acd162(18)
      acd162(20)=acd162(24)+acd162(23)+acd162(22)+acd162(21)+acd162(20)
      acd162(20)=acd162(4)*acd162(20)
      acd162(21)=acd162(3)*acd162(1)
      acd162(22)=-acd162(7)*acd162(6)
      acd162(23)=acd162(9)*acd162(8)
      acd162(24)=-acd162(11)*acd162(10)
      acd162(21)=acd162(24)+acd162(23)+acd162(22)+acd162(21)
      acd162(21)=acd162(2)*acd162(21)
      acd162(20)=acd162(20)+acd162(21)
      brack(ninjaidxt1x0mu0)=acd162(20)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd162h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(62) :: acd162
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd162(1)=dotproduct(ninjaA1,spvak1e1)
      acd162(2)=dotproduct(ninjaE3,spvae1k2)
      acd162(3)=abb162(13)
      acd162(4)=dotproduct(ninjaA1,spvae1k2)
      acd162(5)=dotproduct(ninjaE3,spvak1e1)
      acd162(6)=dotproduct(ninjaE3,spval5e1)
      acd162(7)=abb162(37)
      acd162(8)=dotproduct(ninjaE3,spvae2e1)
      acd162(9)=abb162(23)
      acd162(10)=dotproduct(ninjaE3,spvak2e1)
      acd162(11)=abb162(26)
      acd162(12)=dotproduct(ninjaE3,spval4e1)
      acd162(13)=abb162(43)
      acd162(14)=dotproduct(ninjaA1,spval5e1)
      acd162(15)=dotproduct(ninjaE3,spvae1e2)
      acd162(16)=abb162(17)
      acd162(17)=dotproduct(ninjaE3,spvae1k1)
      acd162(18)=abb162(30)
      acd162(19)=dotproduct(ninjaE3,spvae1l5)
      acd162(20)=abb162(27)
      acd162(21)=dotproduct(ninjaE3,spvae1l4)
      acd162(22)=abb162(39)
      acd162(23)=dotproduct(ninjaA1,spvae1e2)
      acd162(24)=dotproduct(ninjaA1,spvae2e1)
      acd162(25)=dotproduct(ninjaA1,spvae1k1)
      acd162(26)=dotproduct(ninjaA1,spvak2e1)
      acd162(27)=dotproduct(ninjaA1,spvae1l5)
      acd162(28)=dotproduct(ninjaA1,spvae1l4)
      acd162(29)=dotproduct(ninjaA1,spval4e1)
      acd162(30)=dotproduct(ninjaA0,spvak1e1)
      acd162(31)=dotproduct(ninjaA0,spvae1k2)
      acd162(32)=dotproduct(ninjaA0,spval5e1)
      acd162(33)=dotproduct(ninjaA0,spvae1e2)
      acd162(34)=dotproduct(ninjaA0,spvae2e1)
      acd162(35)=dotproduct(ninjaA0,spvae1k1)
      acd162(36)=dotproduct(ninjaA0,spvak2e1)
      acd162(37)=dotproduct(ninjaA0,spvae1l5)
      acd162(38)=dotproduct(ninjaA0,spvae1l4)
      acd162(39)=dotproduct(ninjaA0,spval4e1)
      acd162(40)=abb162(16)
      acd162(41)=abb162(15)
      acd162(42)=abb162(14)
      acd162(43)=abb162(21)
      acd162(44)=abb162(19)
      acd162(45)=abb162(20)
      acd162(46)=abb162(25)
      acd162(47)=abb162(33)
      acd162(48)=abb162(29)
      acd162(49)=abb162(35)
      acd162(50)=acd162(22)*acd162(28)
      acd162(51)=acd162(20)*acd162(27)
      acd162(52)=acd162(18)*acd162(25)
      acd162(53)=acd162(16)*acd162(23)
      acd162(54)=acd162(4)*acd162(7)
      acd162(50)=acd162(54)+acd162(53)+acd162(52)+acd162(50)+acd162(51)
      acd162(50)=acd162(6)*acd162(50)
      acd162(51)=-acd162(13)*acd162(29)
      acd162(52)=acd162(11)*acd162(26)
      acd162(53)=-acd162(9)*acd162(24)
      acd162(54)=acd162(3)*acd162(1)
      acd162(55)=acd162(14)*acd162(7)
      acd162(51)=acd162(55)+acd162(54)+acd162(53)+acd162(51)+acd162(52)
      acd162(51)=acd162(2)*acd162(51)
      acd162(52)=acd162(22)*acd162(21)
      acd162(53)=acd162(20)*acd162(19)
      acd162(54)=acd162(18)*acd162(17)
      acd162(55)=acd162(16)*acd162(15)
      acd162(52)=acd162(52)+acd162(53)+acd162(54)+acd162(55)
      acd162(53)=acd162(14)*acd162(52)
      acd162(54)=acd162(13)*acd162(12)
      acd162(55)=acd162(11)*acd162(10)
      acd162(56)=acd162(9)*acd162(8)
      acd162(57)=acd162(3)*acd162(5)
      acd162(54)=acd162(54)-acd162(55)+acd162(56)-acd162(57)
      acd162(55)=-acd162(4)*acd162(54)
      acd162(50)=acd162(51)+acd162(50)+acd162(53)+acd162(55)
      acd162(51)=acd162(22)*acd162(38)
      acd162(53)=acd162(20)*acd162(37)
      acd162(55)=acd162(18)*acd162(35)
      acd162(56)=acd162(16)*acd162(33)
      acd162(57)=acd162(31)*acd162(7)
      acd162(51)=acd162(57)+acd162(56)+acd162(55)+acd162(53)+acd162(42)+acd162(&
      &51)
      acd162(51)=acd162(6)*acd162(51)
      acd162(53)=-acd162(13)*acd162(39)
      acd162(55)=acd162(11)*acd162(36)
      acd162(56)=-acd162(9)*acd162(34)
      acd162(57)=acd162(3)*acd162(30)
      acd162(58)=acd162(32)*acd162(7)
      acd162(53)=acd162(58)+acd162(57)+acd162(56)+acd162(55)+acd162(41)+acd162(&
      &53)
      acd162(53)=acd162(2)*acd162(53)
      acd162(52)=acd162(32)*acd162(52)
      acd162(54)=-acd162(31)*acd162(54)
      acd162(55)=acd162(21)*acd162(48)
      acd162(56)=acd162(19)*acd162(47)
      acd162(57)=acd162(17)*acd162(45)
      acd162(58)=acd162(15)*acd162(43)
      acd162(59)=acd162(12)*acd162(49)
      acd162(60)=acd162(10)*acd162(46)
      acd162(61)=acd162(8)*acd162(44)
      acd162(62)=acd162(5)*acd162(40)
      acd162(51)=acd162(53)+acd162(51)+acd162(54)+acd162(52)+acd162(62)+acd162(&
      &61)+acd162(60)+acd162(59)+acd162(58)+acd162(57)+acd162(55)+acd162(56)
      brack(ninjaidxt0x0mu0)=acd162(51)
      brack(ninjaidxt0x1mu0)=acd162(50)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d162h4_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd162h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k5
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
end module     p2_gg_httbar_d162h4l132_qp
