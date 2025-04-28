module     p2_gg_httbar_d142h0l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d142h0l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt3mu0 = 0
   integer, parameter :: ninjaidxt2mu0 = 1
   integer, parameter :: ninjaidxt1mu0 = 2
   integer, parameter :: ninjaidxt1mu2 = 3
   integer, parameter :: ninjaidxt0mu0 = 4
   integer, parameter :: ninjaidxt0mu2 = 5
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd142h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(29) :: acd142
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd142(1)=dotproduct(ninjaE3,spvak2e2)
      acd142(2)=dotproduct(ninjaE3,spvae2k2)
      acd142(3)=abb142(16)
      acd142(4)=dotproduct(ninjaE3,spval4e2)
      acd142(5)=abb142(29)
      acd142(6)=dotproduct(ninjaE3,spvae1e2)
      acd142(7)=abb142(18)
      acd142(8)=dotproduct(ninjaE3,spval5e2)
      acd142(9)=abb142(33)
      acd142(10)=dotproduct(ninjaE3,spvak1e2)
      acd142(11)=abb142(21)
      acd142(12)=dotproduct(ninjaE3,spvae2k1)
      acd142(13)=abb142(17)
      acd142(14)=abb142(19)
      acd142(15)=abb142(98)
      acd142(16)=dotproduct(ninjaE3,spvae2e1)
      acd142(17)=abb142(76)
      acd142(18)=dotproduct(ninjaE3,spvae2l5)
      acd142(19)=abb142(79)
      acd142(20)=dotproduct(ninjaE3,spvae2l4)
      acd142(21)=abb142(77)
      acd142(22)=abb142(96)
      acd142(23)=abb142(102)
      acd142(24)=-acd142(20)*acd142(15)
      acd142(25)=acd142(5)*acd142(2)
      acd142(26)=acd142(13)*acd142(12)
      acd142(27)=acd142(17)*acd142(16)
      acd142(28)=acd142(19)*acd142(18)
      acd142(24)=acd142(28)+acd142(27)+acd142(26)+acd142(25)+acd142(24)
      acd142(24)=acd142(4)*acd142(24)
      acd142(25)=-acd142(18)*acd142(15)
      acd142(26)=acd142(9)*acd142(2)
      acd142(27)=acd142(14)*acd142(12)
      acd142(28)=acd142(22)*acd142(16)
      acd142(29)=acd142(23)*acd142(20)
      acd142(25)=acd142(29)+acd142(28)+acd142(27)+acd142(26)+acd142(25)
      acd142(25)=acd142(8)*acd142(25)
      acd142(26)=acd142(3)*acd142(1)
      acd142(27)=acd142(7)*acd142(6)
      acd142(28)=acd142(11)*acd142(10)
      acd142(26)=acd142(28)+acd142(27)+acd142(26)
      acd142(26)=acd142(2)*acd142(26)
      acd142(27)=-acd142(10)*acd142(15)*acd142(12)
      acd142(28)=acd142(21)*acd142(6)*acd142(16)
      acd142(24)=acd142(28)+acd142(27)+acd142(25)+acd142(24)+acd142(26)
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd142(24)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd142h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(85) :: acd142
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd142(1)=dotproduct(ninjaE3,spvae2k2)
      acd142(2)=dotproduct(ninjaE4,spvak2e2)
      acd142(3)=abb142(16)
      acd142(4)=dotproduct(ninjaE4,spval4e2)
      acd142(5)=abb142(29)
      acd142(6)=dotproduct(ninjaE4,spvae1e2)
      acd142(7)=abb142(18)
      acd142(8)=dotproduct(ninjaE4,spval5e2)
      acd142(9)=abb142(33)
      acd142(10)=dotproduct(ninjaE4,spvak1e2)
      acd142(11)=abb142(21)
      acd142(12)=dotproduct(ninjaE3,spvak2e2)
      acd142(13)=dotproduct(ninjaE4,spvae2k2)
      acd142(14)=dotproduct(ninjaE3,spvae2k1)
      acd142(15)=abb142(17)
      acd142(16)=abb142(19)
      acd142(17)=abb142(98)
      acd142(18)=dotproduct(ninjaE3,spval4e2)
      acd142(19)=dotproduct(ninjaE4,spvae2k1)
      acd142(20)=dotproduct(ninjaE4,spvae2e1)
      acd142(21)=abb142(76)
      acd142(22)=dotproduct(ninjaE4,spvae2l4)
      acd142(23)=dotproduct(ninjaE4,spvae2l5)
      acd142(24)=abb142(79)
      acd142(25)=dotproduct(ninjaE3,spvae1e2)
      acd142(26)=abb142(77)
      acd142(27)=dotproduct(ninjaE3,spval5e2)
      acd142(28)=abb142(96)
      acd142(29)=abb142(102)
      acd142(30)=dotproduct(ninjaE3,spvak1e2)
      acd142(31)=dotproduct(ninjaE3,spvae2e1)
      acd142(32)=dotproduct(ninjaE3,spvae2l4)
      acd142(33)=dotproduct(ninjaE3,spvae2l5)
      acd142(34)=abb142(14)
      acd142(35)=dotproduct(ninjaA,ninjaE3)
      acd142(36)=dotproduct(ninjaA,spvae2k2)
      acd142(37)=dotproduct(ninjaA,spvak2e2)
      acd142(38)=dotproduct(ninjaA,spvae2k1)
      acd142(39)=dotproduct(ninjaA,spval4e2)
      acd142(40)=dotproduct(ninjaA,spvae1e2)
      acd142(41)=dotproduct(ninjaA,spval5e2)
      acd142(42)=dotproduct(ninjaA,spvak1e2)
      acd142(43)=dotproduct(ninjaA,spvae2e1)
      acd142(44)=dotproduct(ninjaA,spvae2l4)
      acd142(45)=dotproduct(ninjaA,spvae2l5)
      acd142(46)=abb142(12)
      acd142(47)=abb142(13)
      acd142(48)=abb142(15)
      acd142(49)=abb142(26)
      acd142(50)=abb142(45)
      acd142(51)=abb142(90)
      acd142(52)=dotproduct(ninjaE3,spval3e2)
      acd142(53)=abb142(20)
      acd142(54)=abb142(22)
      acd142(55)=dotproduct(ninjaE3,spvae2l3)
      acd142(56)=abb142(28)
      acd142(57)=abb142(41)
      acd142(58)=abb142(74)
      acd142(59)=abb142(82)
      acd142(60)=dotproduct(ninjaA,ninjaA)
      acd142(61)=dotproduct(ninjaA,spval3e2)
      acd142(62)=dotproduct(ninjaA,spvae2l3)
      acd142(63)=abb142(36)
      acd142(64)=acd142(30)*acd142(19)
      acd142(65)=acd142(14)*acd142(10)
      acd142(66)=acd142(8)*acd142(33)
      acd142(67)=acd142(4)*acd142(32)
      acd142(68)=acd142(27)*acd142(23)
      acd142(69)=acd142(18)*acd142(22)
      acd142(64)=acd142(66)+acd142(67)+acd142(68)+acd142(69)+acd142(64)+acd142(&
      &65)
      acd142(64)=acd142(64)*acd142(17)
      acd142(65)=acd142(11)*acd142(10)
      acd142(66)=acd142(7)*acd142(6)
      acd142(67)=acd142(3)*acd142(2)
      acd142(68)=acd142(8)*acd142(9)
      acd142(69)=acd142(4)*acd142(5)
      acd142(65)=acd142(65)+acd142(68)+acd142(69)+acd142(66)+acd142(67)
      acd142(65)=acd142(65)*acd142(1)
      acd142(66)=acd142(24)*acd142(23)
      acd142(67)=acd142(21)*acd142(20)
      acd142(68)=acd142(15)*acd142(19)
      acd142(69)=acd142(13)*acd142(5)
      acd142(66)=acd142(66)+acd142(67)+acd142(68)+acd142(69)
      acd142(66)=acd142(66)*acd142(18)
      acd142(67)=acd142(29)*acd142(22)
      acd142(68)=acd142(28)*acd142(20)
      acd142(69)=acd142(16)*acd142(19)
      acd142(70)=acd142(13)*acd142(9)
      acd142(67)=acd142(67)+acd142(68)+acd142(69)+acd142(70)
      acd142(67)=acd142(67)*acd142(27)
      acd142(68)=acd142(11)*acd142(30)
      acd142(69)=acd142(7)*acd142(25)
      acd142(70)=acd142(3)*acd142(12)
      acd142(68)=acd142(70)+acd142(68)+acd142(69)
      acd142(69)=acd142(68)*acd142(13)
      acd142(70)=acd142(24)*acd142(33)
      acd142(71)=acd142(31)*acd142(21)
      acd142(72)=acd142(14)*acd142(15)
      acd142(70)=acd142(72)+acd142(70)+acd142(71)
      acd142(71)=acd142(70)*acd142(4)
      acd142(72)=acd142(29)*acd142(32)
      acd142(73)=acd142(31)*acd142(28)
      acd142(74)=acd142(14)*acd142(16)
      acd142(72)=acd142(74)+acd142(72)+acd142(73)
      acd142(73)=acd142(72)*acd142(8)
      acd142(74)=acd142(25)*acd142(20)
      acd142(75)=acd142(31)*acd142(6)
      acd142(74)=acd142(74)+acd142(75)
      acd142(74)=acd142(74)*acd142(26)
      acd142(64)=-acd142(64)+acd142(65)+acd142(74)+acd142(69)+acd142(71)+acd142&
      &(73)+acd142(34)+acd142(66)+acd142(67)
      acd142(65)=-acd142(30)*acd142(38)
      acd142(66)=-acd142(14)*acd142(42)
      acd142(67)=-acd142(41)*acd142(33)
      acd142(69)=-acd142(39)*acd142(32)
      acd142(71)=-acd142(27)*acd142(45)
      acd142(73)=-acd142(18)*acd142(44)
      acd142(65)=acd142(73)+acd142(71)+acd142(69)+acd142(67)+acd142(65)+acd142(&
      &66)
      acd142(65)=acd142(17)*acd142(65)
      acd142(66)=acd142(11)*acd142(42)
      acd142(67)=acd142(7)*acd142(40)
      acd142(69)=acd142(3)*acd142(37)
      acd142(71)=acd142(41)*acd142(9)
      acd142(73)=acd142(39)*acd142(5)
      acd142(66)=acd142(46)+acd142(66)+acd142(71)+acd142(73)+acd142(67)+acd142(&
      &69)
      acd142(67)=acd142(1)*acd142(66)
      acd142(69)=acd142(41)*acd142(72)
      acd142(70)=acd142(39)*acd142(70)
      acd142(68)=acd142(36)*acd142(68)
      acd142(71)=acd142(29)*acd142(44)
      acd142(72)=acd142(28)*acd142(43)
      acd142(73)=acd142(16)*acd142(38)
      acd142(71)=acd142(71)+acd142(72)+acd142(73)+acd142(51)
      acd142(72)=acd142(36)*acd142(9)
      acd142(72)=acd142(72)+acd142(71)
      acd142(72)=acd142(27)*acd142(72)
      acd142(73)=acd142(24)*acd142(45)
      acd142(74)=acd142(21)*acd142(43)
      acd142(75)=acd142(15)*acd142(38)
      acd142(73)=acd142(73)+acd142(74)+acd142(75)+acd142(49)
      acd142(74)=acd142(36)*acd142(5)
      acd142(74)=acd142(74)+acd142(73)
      acd142(74)=acd142(18)*acd142(74)
      acd142(75)=acd142(56)*acd142(55)
      acd142(76)=acd142(53)*acd142(52)
      acd142(77)=acd142(34)*acd142(35)
      acd142(78)=acd142(12)*acd142(47)
      acd142(79)=acd142(33)*acd142(59)
      acd142(80)=acd142(32)*acd142(58)
      acd142(81)=acd142(30)*acd142(54)
      acd142(82)=acd142(26)*acd142(43)
      acd142(82)=acd142(50)+acd142(82)
      acd142(82)=acd142(25)*acd142(82)
      acd142(83)=acd142(26)*acd142(40)
      acd142(83)=acd142(83)+acd142(57)
      acd142(84)=acd142(31)*acd142(83)
      acd142(85)=acd142(14)*acd142(48)
      acd142(65)=acd142(65)+acd142(67)+acd142(74)+acd142(72)+acd142(68)+acd142(&
      &70)+acd142(69)+acd142(85)+acd142(84)+acd142(82)+acd142(81)+acd142(80)+ac&
      &d142(79)+acd142(78)+2.0_ki*acd142(77)+acd142(75)+acd142(76)
      acd142(67)=ninjaP*acd142(64)
      acd142(66)=acd142(36)*acd142(66)
      acd142(68)=acd142(41)*acd142(71)
      acd142(69)=acd142(39)*acd142(73)
      acd142(70)=-acd142(38)*acd142(42)
      acd142(71)=-acd142(41)*acd142(45)
      acd142(72)=-acd142(39)*acd142(44)
      acd142(70)=acd142(72)+acd142(70)+acd142(71)
      acd142(70)=acd142(17)*acd142(70)
      acd142(71)=acd142(43)*acd142(83)
      acd142(72)=acd142(56)*acd142(62)
      acd142(73)=acd142(53)*acd142(61)
      acd142(74)=acd142(37)*acd142(47)
      acd142(75)=acd142(34)*acd142(60)
      acd142(76)=acd142(45)*acd142(59)
      acd142(77)=acd142(44)*acd142(58)
      acd142(78)=acd142(42)*acd142(54)
      acd142(79)=acd142(40)*acd142(50)
      acd142(80)=acd142(38)*acd142(48)
      acd142(66)=acd142(67)+acd142(70)+acd142(66)+acd142(69)+acd142(68)+acd142(&
      &80)+acd142(79)+acd142(78)+acd142(77)+acd142(76)+acd142(75)+acd142(74)+ac&
      &d142(73)+acd142(63)+acd142(72)+acd142(71)
      brack(ninjaidxt1mu0)=acd142(65)
      brack(ninjaidxt1mu2)=0.0_ki
      brack(ninjaidxt0mu0)=acd142(66)
      brack(ninjaidxt0mu2)=acd142(64)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d142h0_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd142h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3-k5
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d142h0l131_qp
