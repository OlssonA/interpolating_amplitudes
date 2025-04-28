module     p2_gg_httbar_d172h12l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d172h12l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd172h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(23) :: acd172
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd172(1)=dotproduct(ninjaE3,spvak2e1)
      acd172(2)=dotproduct(ninjaE3,spvae1e2)
      acd172(3)=abb172(16)
      acd172(4)=dotproduct(ninjaE3,spvae1l5)
      acd172(5)=abb172(17)
      acd172(6)=dotproduct(ninjaE3,spvae1l4)
      acd172(7)=abb172(18)
      acd172(8)=dotproduct(ninjaE3,spvae1k2)
      acd172(9)=abb172(19)
      acd172(10)=dotproduct(ninjaE3,spvae2e1)
      acd172(11)=abb172(54)
      acd172(12)=abb172(39)
      acd172(13)=dotproduct(ninjaE3,spval5e1)
      acd172(14)=abb172(86)
      acd172(15)=dotproduct(ninjaE3,spval4e1)
      acd172(16)=abb172(57)
      acd172(17)=abb172(23)
      acd172(18)=abb172(24)
      acd172(19)=acd172(14)*acd172(13)
      acd172(20)=acd172(5)*acd172(1)
      acd172(21)=acd172(12)*acd172(10)
      acd172(22)=acd172(16)*acd172(15)
      acd172(19)=acd172(22)+acd172(21)+acd172(20)+acd172(19)
      acd172(19)=acd172(4)*acd172(19)
      acd172(20)=acd172(15)*acd172(14)
      acd172(21)=acd172(7)*acd172(1)
      acd172(22)=acd172(17)*acd172(10)
      acd172(23)=acd172(18)*acd172(13)
      acd172(20)=acd172(23)+acd172(22)+acd172(21)+acd172(20)
      acd172(20)=acd172(6)*acd172(20)
      acd172(21)=acd172(3)*acd172(1)
      acd172(22)=acd172(11)*acd172(10)
      acd172(21)=acd172(22)+acd172(21)
      acd172(21)=acd172(2)*acd172(21)
      acd172(22)=acd172(9)*acd172(8)*acd172(1)
      acd172(19)=acd172(22)+acd172(20)+acd172(19)+acd172(21)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd172(19)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd172h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(68) :: acd172
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd172(1)=dotproduct(ninjaA1,spvae1k2)
      acd172(2)=dotproduct(ninjaE3,spvak2e1)
      acd172(3)=abb172(19)
      acd172(4)=dotproduct(ninjaA1,spvak2e1)
      acd172(5)=dotproduct(ninjaE3,spvae1k2)
      acd172(6)=dotproduct(ninjaE3,spvae1e2)
      acd172(7)=abb172(16)
      acd172(8)=dotproduct(ninjaE3,spvae1l5)
      acd172(9)=abb172(17)
      acd172(10)=dotproduct(ninjaE3,spvae1l4)
      acd172(11)=abb172(18)
      acd172(12)=dotproduct(ninjaA1,spvae1e2)
      acd172(13)=dotproduct(ninjaE3,spvae2e1)
      acd172(14)=abb172(54)
      acd172(15)=dotproduct(ninjaA1,spvae1l5)
      acd172(16)=abb172(39)
      acd172(17)=dotproduct(ninjaE3,spval5e1)
      acd172(18)=abb172(86)
      acd172(19)=dotproduct(ninjaE3,spval4e1)
      acd172(20)=abb172(57)
      acd172(21)=dotproduct(ninjaA1,spvae1l4)
      acd172(22)=abb172(23)
      acd172(23)=abb172(24)
      acd172(24)=dotproduct(ninjaA1,spvae2e1)
      acd172(25)=dotproduct(ninjaA1,spval5e1)
      acd172(26)=dotproduct(ninjaA1,spval4e1)
      acd172(27)=dotproduct(ninjaA0,ninjaE3)
      acd172(28)=abb172(13)
      acd172(29)=dotproduct(ninjaA0,spvae1k2)
      acd172(30)=dotproduct(ninjaA0,spvak2e1)
      acd172(31)=dotproduct(ninjaA0,spvae1e2)
      acd172(32)=dotproduct(ninjaA0,spvae1l5)
      acd172(33)=dotproduct(ninjaA0,spvae1l4)
      acd172(34)=dotproduct(ninjaA0,spvae2e1)
      acd172(35)=dotproduct(ninjaA0,spval5e1)
      acd172(36)=dotproduct(ninjaA0,spval4e1)
      acd172(37)=abb172(14)
      acd172(38)=abb172(15)
      acd172(39)=abb172(33)
      acd172(40)=abb172(38)
      acd172(41)=abb172(22)
      acd172(42)=dotproduct(ninjaE3,spvae1l3)
      acd172(43)=abb172(21)
      acd172(44)=abb172(31)
      acd172(45)=abb172(56)
      acd172(46)=abb172(25)
      acd172(47)=dotproduct(ninjaE3,spval3e1)
      acd172(48)=abb172(27)
      acd172(49)=acd172(17)*acd172(18)
      acd172(50)=acd172(9)*acd172(2)
      acd172(51)=acd172(16)*acd172(13)
      acd172(52)=acd172(20)*acd172(19)
      acd172(49)=acd172(49)+acd172(50)+acd172(51)+acd172(52)
      acd172(50)=acd172(15)*acd172(49)
      acd172(51)=acd172(19)*acd172(18)
      acd172(52)=acd172(11)*acd172(2)
      acd172(53)=acd172(22)*acd172(13)
      acd172(54)=acd172(23)*acd172(17)
      acd172(51)=acd172(51)+acd172(52)+acd172(53)+acd172(54)
      acd172(52)=acd172(21)*acd172(51)
      acd172(53)=acd172(7)*acd172(6)
      acd172(54)=acd172(9)*acd172(8)
      acd172(55)=acd172(11)*acd172(10)
      acd172(56)=acd172(5)*acd172(3)
      acd172(53)=acd172(56)+acd172(55)+acd172(53)+acd172(54)
      acd172(54)=acd172(4)*acd172(53)
      acd172(55)=acd172(14)*acd172(6)
      acd172(56)=acd172(16)*acd172(8)
      acd172(57)=acd172(22)*acd172(10)
      acd172(55)=acd172(57)+acd172(55)+acd172(56)
      acd172(56)=acd172(24)*acd172(55)
      acd172(57)=acd172(7)*acd172(2)
      acd172(58)=acd172(14)*acd172(13)
      acd172(57)=acd172(57)+acd172(58)
      acd172(58)=acd172(12)*acd172(57)
      acd172(59)=acd172(18)*acd172(8)
      acd172(60)=acd172(23)*acd172(10)
      acd172(59)=acd172(59)+acd172(60)
      acd172(60)=acd172(25)*acd172(59)
      acd172(61)=acd172(18)*acd172(10)
      acd172(62)=acd172(20)*acd172(8)
      acd172(61)=acd172(61)+acd172(62)
      acd172(62)=acd172(26)*acd172(61)
      acd172(63)=acd172(3)*acd172(2)
      acd172(64)=acd172(1)*acd172(63)
      acd172(50)=acd172(64)+acd172(62)+acd172(60)+acd172(58)+acd172(56)+acd172(&
      &54)+acd172(52)+acd172(50)
      acd172(52)=acd172(30)*acd172(53)
      acd172(49)=acd172(32)*acd172(49)
      acd172(51)=acd172(33)*acd172(51)
      acd172(53)=acd172(34)*acd172(55)
      acd172(54)=acd172(31)*acd172(57)
      acd172(55)=acd172(35)*acd172(59)
      acd172(56)=acd172(36)*acd172(61)
      acd172(57)=acd172(28)*acd172(27)
      acd172(58)=acd172(29)*acd172(63)
      acd172(59)=acd172(37)*acd172(5)
      acd172(60)=acd172(38)*acd172(2)
      acd172(61)=acd172(39)*acd172(6)
      acd172(62)=acd172(40)*acd172(8)
      acd172(63)=acd172(41)*acd172(10)
      acd172(64)=acd172(43)*acd172(42)
      acd172(65)=acd172(44)*acd172(13)
      acd172(66)=acd172(45)*acd172(17)
      acd172(67)=acd172(46)*acd172(19)
      acd172(68)=acd172(48)*acd172(47)
      acd172(49)=acd172(68)+acd172(67)+acd172(66)+acd172(65)+acd172(64)+acd172(&
      &63)+acd172(62)+acd172(61)+acd172(60)+acd172(59)+acd172(58)+2.0_ki*acd172&
      &(57)+acd172(56)+acd172(55)+acd172(54)+acd172(53)+acd172(51)+acd172(49)+a&
      &cd172(52)
      brack(ninjaidxt0x0mu0)=acd172(49)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd172(50)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d172h12_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd172h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3-k5
      vecA0(1:4) = + a0(0:3) - qshift(1:4)
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
end module     p2_gg_httbar_d172h12l132_qp
