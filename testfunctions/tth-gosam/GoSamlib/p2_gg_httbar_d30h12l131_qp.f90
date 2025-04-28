module     p2_gg_httbar_d30h12l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d30h12l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2mu0 = 0
   integer, parameter :: ninjaidxt1mu0 = 1
   integer, parameter :: ninjaidxt0mu0 = 2
   integer, parameter :: ninjaidxt0mu2 = 3
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd30h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(31) :: acd30
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd30(1)=dotproduct(ninjaE3,spvae2l4)
      acd30(2)=dotproduct(ninjaE3,spval5e2)
      acd30(3)=abb30(13)
      acd30(4)=dotproduct(ninjaE3,spvak1e2)
      acd30(5)=abb30(63)
      acd30(6)=dotproduct(ninjaE3,spvae1e2)
      acd30(7)=abb30(46)
      acd30(8)=dotproduct(ninjaE3,spvae2l3)
      acd30(9)=abb30(30)
      acd30(10)=dotproduct(ninjaE3,spvak2e2)
      acd30(11)=dotproduct(ninjaE3,spvae2e1)
      acd30(12)=abb30(15)
      acd30(13)=dotproduct(ninjaE3,spvae2k2)
      acd30(14)=abb30(19)
      acd30(15)=dotproduct(ninjaE3,spvae2k1)
      acd30(16)=abb30(22)
      acd30(17)=dotproduct(ninjaE3,spvae2l5)
      acd30(18)=abb30(24)
      acd30(19)=dotproduct(ninjaE3,spval3e2)
      acd30(20)=abb30(34)
      acd30(21)=abb30(28)
      acd30(22)=abb30(21)
      acd30(23)=abb30(48)
      acd30(24)=abb30(45)
      acd30(25)=abb30(50)
      acd30(26)=acd30(12)*acd30(11)
      acd30(27)=acd30(14)*acd30(13)
      acd30(28)=acd30(16)*acd30(15)
      acd30(29)=acd30(18)*acd30(17)
      acd30(26)=acd30(29)+acd30(28)+acd30(27)+acd30(26)
      acd30(26)=acd30(10)*acd30(26)
      acd30(27)=acd30(20)*acd30(11)
      acd30(28)=acd30(21)*acd30(13)
      acd30(29)=acd30(22)*acd30(15)
      acd30(30)=acd30(23)*acd30(17)
      acd30(27)=acd30(30)+acd30(29)+acd30(28)+acd30(27)
      acd30(27)=acd30(19)*acd30(27)
      acd30(28)=acd30(3)*acd30(2)
      acd30(29)=acd30(5)*acd30(4)
      acd30(30)=-acd30(7)*acd30(6)
      acd30(28)=acd30(30)+acd30(28)+acd30(29)
      acd30(28)=acd30(1)*acd30(28)
      acd30(29)=acd30(9)*acd30(2)
      acd30(30)=acd30(24)*acd30(4)
      acd30(31)=-acd30(25)*acd30(6)
      acd30(29)=acd30(31)+acd30(30)+acd30(29)
      acd30(29)=acd30(8)*acd30(29)
      acd30(26)=acd30(27)+acd30(26)+acd30(29)+acd30(28)
      brack(ninjaidxt2mu0)=acd30(26)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd30h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(79) :: acd30
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd30(1)=dotproduct(ninjaE3,spvae2l4)
      acd30(2)=dotproduct(ninjaE4,spval5e2)
      acd30(3)=abb30(13)
      acd30(4)=dotproduct(ninjaE4,spvak1e2)
      acd30(5)=abb30(63)
      acd30(6)=dotproduct(ninjaE4,spvae1e2)
      acd30(7)=abb30(46)
      acd30(8)=dotproduct(ninjaE3,spvak2e2)
      acd30(9)=dotproduct(ninjaE4,spvae2e1)
      acd30(10)=abb30(15)
      acd30(11)=dotproduct(ninjaE4,spvae2k2)
      acd30(12)=abb30(19)
      acd30(13)=dotproduct(ninjaE4,spvae2k1)
      acd30(14)=abb30(22)
      acd30(15)=dotproduct(ninjaE4,spvae2l5)
      acd30(16)=abb30(24)
      acd30(17)=dotproduct(ninjaE3,spval5e2)
      acd30(18)=dotproduct(ninjaE4,spvae2l4)
      acd30(19)=dotproduct(ninjaE4,spvae2l3)
      acd30(20)=abb30(30)
      acd30(21)=dotproduct(ninjaE3,spvae2e1)
      acd30(22)=dotproduct(ninjaE4,spvak2e2)
      acd30(23)=dotproduct(ninjaE4,spval3e2)
      acd30(24)=abb30(34)
      acd30(25)=dotproduct(ninjaE3,spvak1e2)
      acd30(26)=abb30(45)
      acd30(27)=dotproduct(ninjaE3,spvae2l3)
      acd30(28)=abb30(50)
      acd30(29)=dotproduct(ninjaE3,spvae2k2)
      acd30(30)=abb30(28)
      acd30(31)=dotproduct(ninjaE3,spvae2k1)
      acd30(32)=abb30(21)
      acd30(33)=dotproduct(ninjaE3,spval3e2)
      acd30(34)=abb30(48)
      acd30(35)=dotproduct(ninjaE3,spvae2l5)
      acd30(36)=dotproduct(ninjaE3,spvae1e2)
      acd30(37)=dotproduct(ninjaA,spvae2l4)
      acd30(38)=dotproduct(ninjaA,spvak2e2)
      acd30(39)=dotproduct(ninjaA,spval5e2)
      acd30(40)=dotproduct(ninjaA,spvae2e1)
      acd30(41)=dotproduct(ninjaA,spvak1e2)
      acd30(42)=dotproduct(ninjaA,spvae2l3)
      acd30(43)=dotproduct(ninjaA,spvae2k2)
      acd30(44)=dotproduct(ninjaA,spvae2k1)
      acd30(45)=dotproduct(ninjaA,spval3e2)
      acd30(46)=dotproduct(ninjaA,spvae2l5)
      acd30(47)=dotproduct(ninjaA,spvae1e2)
      acd30(48)=abb30(11)
      acd30(49)=abb30(12)
      acd30(50)=abb30(43)
      acd30(51)=abb30(14)
      acd30(52)=abb30(16)
      acd30(53)=abb30(18)
      acd30(54)=abb30(36)
      acd30(55)=abb30(20)
      acd30(56)=abb30(37)
      acd30(57)=abb30(41)
      acd30(58)=abb30(25)
      acd30(59)=abb30(10)
      acd30(60)=acd30(34)*acd30(15)
      acd30(61)=acd30(32)*acd30(13)
      acd30(62)=acd30(30)*acd30(11)
      acd30(63)=acd30(24)*acd30(9)
      acd30(60)=acd30(60)+acd30(61)+acd30(62)+acd30(63)
      acd30(60)=acd30(60)*acd30(33)
      acd30(61)=acd30(16)*acd30(15)
      acd30(62)=acd30(14)*acd30(13)
      acd30(63)=acd30(12)*acd30(11)
      acd30(64)=acd30(10)*acd30(9)
      acd30(61)=acd30(64)+acd30(61)+acd30(62)+acd30(63)
      acd30(61)=acd30(61)*acd30(8)
      acd30(62)=acd30(34)*acd30(35)
      acd30(63)=acd30(31)*acd30(32)
      acd30(64)=acd30(29)*acd30(30)
      acd30(65)=acd30(21)*acd30(24)
      acd30(62)=acd30(63)+acd30(62)+acd30(64)+acd30(65)
      acd30(63)=acd30(62)*acd30(23)
      acd30(64)=acd30(16)*acd30(35)
      acd30(65)=acd30(14)*acd30(31)
      acd30(66)=acd30(12)*acd30(29)
      acd30(67)=acd30(10)*acd30(21)
      acd30(64)=acd30(64)+acd30(65)+acd30(66)+acd30(67)
      acd30(65)=acd30(64)*acd30(22)
      acd30(66)=acd30(25)*acd30(26)
      acd30(67)=acd30(17)*acd30(20)
      acd30(68)=acd30(28)*acd30(36)
      acd30(66)=-acd30(68)+acd30(66)+acd30(67)
      acd30(66)=acd30(66)*acd30(19)
      acd30(67)=acd30(28)*acd30(6)
      acd30(69)=acd30(26)*acd30(4)
      acd30(70)=acd30(20)*acd30(2)
      acd30(67)=-acd30(70)+acd30(67)-acd30(69)
      acd30(67)=acd30(67)*acd30(27)
      acd30(69)=acd30(7)*acd30(6)
      acd30(70)=acd30(5)*acd30(4)
      acd30(71)=acd30(3)*acd30(2)
      acd30(69)=-acd30(71)+acd30(69)-acd30(70)
      acd30(69)=acd30(69)*acd30(1)
      acd30(70)=acd30(7)*acd30(36)
      acd30(71)=acd30(5)*acd30(25)
      acd30(72)=acd30(3)*acd30(17)
      acd30(70)=-acd30(72)+acd30(70)-acd30(71)
      acd30(71)=acd30(70)*acd30(18)
      acd30(60)=-acd30(61)-acd30(66)+acd30(69)-acd30(63)+acd30(71)-acd30(65)+ac&
      &d30(67)-acd30(60)
      acd30(61)=acd30(45)*acd30(62)
      acd30(62)=acd30(38)*acd30(64)
      acd30(63)=acd30(34)*acd30(46)
      acd30(64)=acd30(32)*acd30(44)
      acd30(65)=acd30(30)*acd30(43)
      acd30(66)=acd30(24)*acd30(40)
      acd30(63)=acd30(56)+acd30(63)+acd30(64)+acd30(65)+acd30(66)
      acd30(64)=acd30(33)*acd30(63)
      acd30(65)=acd30(16)*acd30(46)
      acd30(66)=acd30(14)*acd30(44)
      acd30(67)=acd30(12)*acd30(43)
      acd30(69)=acd30(10)*acd30(40)
      acd30(65)=acd30(65)+acd30(66)+acd30(67)+acd30(69)+acd30(49)
      acd30(66)=acd30(8)*acd30(65)
      acd30(67)=-acd30(37)*acd30(70)
      acd30(69)=acd30(28)*acd30(47)
      acd30(70)=acd30(26)*acd30(41)
      acd30(71)=acd30(20)*acd30(39)
      acd30(69)=-acd30(69)+acd30(70)+acd30(71)+acd30(53)
      acd30(70)=acd30(27)*acd30(69)
      acd30(71)=acd30(7)*acd30(47)
      acd30(72)=acd30(5)*acd30(41)
      acd30(73)=acd30(3)*acd30(39)
      acd30(71)=-acd30(71)+acd30(72)+acd30(73)+acd30(48)
      acd30(72)=acd30(1)*acd30(71)
      acd30(73)=acd30(36)*acd30(58)
      acd30(74)=acd30(35)*acd30(57)
      acd30(75)=acd30(31)*acd30(55)
      acd30(76)=acd30(29)*acd30(54)
      acd30(68)=-acd30(42)*acd30(68)
      acd30(77)=acd30(26)*acd30(42)
      acd30(77)=acd30(52)+acd30(77)
      acd30(77)=acd30(25)*acd30(77)
      acd30(78)=acd30(21)*acd30(51)
      acd30(79)=acd30(20)*acd30(42)
      acd30(79)=acd30(50)+acd30(79)
      acd30(79)=acd30(17)*acd30(79)
      acd30(61)=acd30(66)+acd30(64)+acd30(72)+acd30(70)+acd30(62)+acd30(61)+acd&
      &30(79)+acd30(78)+acd30(77)+acd30(68)+acd30(76)+acd30(75)+acd30(73)+acd30&
      &(74)+acd30(67)
      acd30(62)=-ninjaP*acd30(60)
      acd30(63)=acd30(45)*acd30(63)
      acd30(64)=acd30(38)*acd30(65)
      acd30(65)=acd30(42)*acd30(69)
      acd30(66)=acd30(37)*acd30(71)
      acd30(67)=acd30(47)*acd30(58)
      acd30(68)=acd30(46)*acd30(57)
      acd30(69)=acd30(44)*acd30(55)
      acd30(70)=acd30(43)*acd30(54)
      acd30(71)=acd30(41)*acd30(52)
      acd30(72)=acd30(40)*acd30(51)
      acd30(73)=acd30(39)*acd30(50)
      acd30(62)=acd30(62)+acd30(64)+acd30(63)+acd30(73)+acd30(72)+acd30(71)+acd&
      &30(70)+acd30(69)+acd30(68)+acd30(59)+acd30(67)+acd30(66)+acd30(65)
      brack(ninjaidxt1mu0)=acd30(61)
      brack(ninjaidxt0mu0)=acd30(62)
      brack(ninjaidxt0mu2)=-acd30(60)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d30h12_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd30h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k2+k3+k4
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d30h12l131_qp
