module     p2_gg_httbar_d89h0l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d89h0l131_qp.f90
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
      use p2_gg_httbar_abbrevd89h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(15) :: acd89
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd89(1)=dotproduct(ninjaE3,spvae2k2)
      acd89(2)=dotproduct(ninjaE3,spval4e1)
      acd89(3)=dotproduct(ninjaE3,spvae1e2)
      acd89(4)=abb89(9)
      acd89(5)=dotproduct(ninjaE3,spvae2l3)
      acd89(6)=abb89(44)
      acd89(7)=dotproduct(ninjaE3,spvae1k2)
      acd89(8)=dotproduct(ninjaE3,spval5e2)
      acd89(9)=dotproduct(ninjaE3,spvae2e1)
      acd89(10)=abb89(24)
      acd89(11)=dotproduct(ninjaE3,spval3e2)
      acd89(12)=abb89(30)
      acd89(13)=acd89(10)*acd89(8)
      acd89(14)=acd89(12)*acd89(11)
      acd89(13)=acd89(14)+acd89(13)
      acd89(13)=acd89(13)*acd89(9)*acd89(7)
      acd89(14)=acd89(4)*acd89(1)
      acd89(15)=-acd89(6)*acd89(5)
      acd89(14)=acd89(14)+acd89(15)
      acd89(14)=acd89(14)*acd89(3)*acd89(2)
      acd89(13)=acd89(14)+acd89(13)
      brack(ninjaidxt2mu0)=acd89(13)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd89h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(83) :: acd89
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd89(1)=dotproduct(ninjaE3,spvae1e2)
      acd89(2)=dotproduct(ninjaE3,spvae2k2)
      acd89(3)=dotproduct(ninjaE4,spval4e1)
      acd89(4)=abb89(9)
      acd89(5)=dotproduct(ninjaE3,spval4e1)
      acd89(6)=dotproduct(ninjaE4,spvae2k2)
      acd89(7)=dotproduct(ninjaE4,spvae2l3)
      acd89(8)=abb89(44)
      acd89(9)=dotproduct(ninjaE3,spvae2l3)
      acd89(10)=dotproduct(ninjaE4,spvae1e2)
      acd89(11)=dotproduct(ninjaE3,spval5e2)
      acd89(12)=dotproduct(ninjaE3,spvae2e1)
      acd89(13)=dotproduct(ninjaE4,spvae1k2)
      acd89(14)=abb89(24)
      acd89(15)=dotproduct(ninjaE3,spvae1k2)
      acd89(16)=dotproduct(ninjaE4,spvae2e1)
      acd89(17)=dotproduct(ninjaE4,spval5e2)
      acd89(18)=dotproduct(ninjaE4,spval3e2)
      acd89(19)=abb89(30)
      acd89(20)=dotproduct(ninjaE3,spval3e2)
      acd89(21)=dotproduct(ninjaA,spvae1e2)
      acd89(22)=dotproduct(ninjaA,spvae2k2)
      acd89(23)=dotproduct(ninjaA,spval4e1)
      acd89(24)=dotproduct(ninjaA,spval5e2)
      acd89(25)=dotproduct(ninjaA,spvae2e1)
      acd89(26)=dotproduct(ninjaA,spvae1k2)
      acd89(27)=dotproduct(ninjaA,spval3e2)
      acd89(28)=dotproduct(ninjaA,spvae2l3)
      acd89(29)=abb89(11)
      acd89(30)=abb89(21)
      acd89(31)=abb89(36)
      acd89(32)=abb89(12)
      acd89(33)=abb89(19)
      acd89(34)=abb89(17)
      acd89(35)=dotproduct(ninjaE3,spval4e2)
      acd89(36)=abb89(26)
      acd89(37)=dotproduct(k2,ninjaE3)
      acd89(38)=abb89(23)
      acd89(39)=dotproduct(ninjaA,ninjaE3)
      acd89(40)=abb89(25)
      acd89(41)=dotproduct(ninjaA,spval4e2)
      acd89(42)=abb89(8)
      acd89(43)=abb89(10)
      acd89(44)=abb89(33)
      acd89(45)=abb89(46)
      acd89(46)=abb89(31)
      acd89(47)=abb89(13)
      acd89(48)=dotproduct(ninjaE3,spval5k2)
      acd89(49)=abb89(14)
      acd89(50)=abb89(16)
      acd89(51)=dotproduct(ninjaE3,spval3k2)
      acd89(52)=abb89(18)
      acd89(53)=abb89(20)
      acd89(54)=abb89(41)
      acd89(55)=dotproduct(ninjaE3,spvae2k1)
      acd89(56)=abb89(27)
      acd89(57)=dotproduct(ninjaE3,spval4k2)
      acd89(58)=abb89(28)
      acd89(59)=dotproduct(ninjaE3,spvak1e2)
      acd89(60)=abb89(29)
      acd89(61)=dotproduct(ninjaE3,spvak2l3)
      acd89(62)=abb89(54)
      acd89(63)=acd89(8)*acd89(7)
      acd89(64)=acd89(4)*acd89(6)
      acd89(63)=acd89(63)-acd89(64)
      acd89(63)=acd89(63)*acd89(5)
      acd89(64)=acd89(8)*acd89(9)
      acd89(65)=acd89(4)*acd89(2)
      acd89(64)=acd89(64)-acd89(65)
      acd89(65)=-acd89(3)*acd89(64)
      acd89(65)=-acd89(63)+acd89(65)
      acd89(65)=acd89(1)*acd89(65)
      acd89(66)=acd89(19)*acd89(18)
      acd89(67)=acd89(14)*acd89(17)
      acd89(66)=acd89(66)+acd89(67)
      acd89(66)=acd89(66)*acd89(15)
      acd89(67)=acd89(19)*acd89(20)
      acd89(68)=acd89(14)*acd89(11)
      acd89(67)=acd89(67)+acd89(68)
      acd89(68)=acd89(13)*acd89(67)
      acd89(68)=acd89(66)+acd89(68)
      acd89(68)=acd89(12)*acd89(68)
      acd89(69)=acd89(67)*acd89(15)
      acd89(70)=acd89(16)*acd89(69)
      acd89(71)=acd89(64)*acd89(5)
      acd89(72)=-acd89(10)*acd89(71)
      acd89(65)=acd89(68)+acd89(65)+acd89(70)+acd89(72)
      acd89(64)=acd89(64)*acd89(23)
      acd89(68)=-acd89(8)*acd89(28)
      acd89(70)=acd89(4)*acd89(22)
      acd89(68)=acd89(70)-acd89(30)+acd89(68)
      acd89(68)=acd89(5)*acd89(68)
      acd89(70)=acd89(9)*acd89(31)
      acd89(72)=acd89(2)*acd89(29)
      acd89(68)=acd89(68)+acd89(70)+acd89(72)-acd89(64)
      acd89(68)=acd89(1)*acd89(68)
      acd89(67)=acd89(67)*acd89(26)
      acd89(70)=acd89(35)*acd89(36)
      acd89(67)=acd89(70)+acd89(67)
      acd89(70)=acd89(19)*acd89(27)
      acd89(72)=acd89(14)*acd89(24)
      acd89(70)=acd89(72)+acd89(33)+acd89(70)
      acd89(70)=acd89(15)*acd89(70)
      acd89(72)=acd89(20)*acd89(34)
      acd89(73)=acd89(11)*acd89(32)
      acd89(70)=acd89(70)+acd89(73)+acd89(72)+acd89(67)
      acd89(70)=acd89(12)*acd89(70)
      acd89(69)=acd89(25)*acd89(69)
      acd89(71)=-acd89(21)*acd89(71)
      acd89(68)=acd89(70)+acd89(68)+acd89(69)+acd89(71)
      acd89(69)=acd89(26)*acd89(27)
      acd89(70)=ninjaP*acd89(20)
      acd89(71)=acd89(13)*acd89(70)
      acd89(69)=acd89(69)+acd89(71)
      acd89(69)=acd89(19)*acd89(69)
      acd89(71)=acd89(26)*acd89(24)
      acd89(72)=ninjaP*acd89(11)
      acd89(73)=acd89(13)*acd89(72)
      acd89(71)=acd89(71)+acd89(73)
      acd89(71)=acd89(14)*acd89(71)
      acd89(66)=ninjaP*acd89(66)
      acd89(73)=acd89(36)*acd89(41)
      acd89(74)=acd89(27)*acd89(34)
      acd89(75)=acd89(24)*acd89(32)
      acd89(76)=acd89(26)*acd89(33)
      acd89(66)=acd89(66)+acd89(71)+acd89(69)+acd89(76)+acd89(75)+acd89(74)+acd&
      &89(46)+acd89(73)
      acd89(66)=acd89(12)*acd89(66)
      acd89(69)=-acd89(23)*acd89(28)
      acd89(71)=ninjaP*acd89(9)
      acd89(73)=-acd89(3)*acd89(71)
      acd89(69)=acd89(69)+acd89(73)
      acd89(69)=acd89(8)*acd89(69)
      acd89(73)=acd89(23)*acd89(22)
      acd89(74)=ninjaP*acd89(2)
      acd89(75)=acd89(3)*acd89(74)
      acd89(73)=acd89(73)+acd89(75)
      acd89(73)=acd89(4)*acd89(73)
      acd89(63)=-ninjaP*acd89(63)
      acd89(75)=acd89(28)*acd89(31)
      acd89(76)=acd89(22)*acd89(29)
      acd89(77)=-acd89(23)*acd89(30)
      acd89(63)=acd89(63)+acd89(73)+acd89(69)+acd89(77)+acd89(76)+acd89(42)+acd&
      &89(75)
      acd89(63)=acd89(1)*acd89(63)
      acd89(69)=acd89(25)*acd89(27)
      acd89(70)=acd89(16)*acd89(70)
      acd89(69)=acd89(69)+acd89(70)
      acd89(69)=acd89(19)*acd89(69)
      acd89(70)=acd89(25)*acd89(24)
      acd89(72)=acd89(16)*acd89(72)
      acd89(70)=acd89(70)+acd89(72)
      acd89(70)=acd89(14)*acd89(70)
      acd89(72)=acd89(25)*acd89(33)
      acd89(69)=acd89(70)+acd89(69)+acd89(47)+acd89(72)
      acd89(69)=acd89(15)*acd89(69)
      acd89(70)=-acd89(21)*acd89(28)
      acd89(71)=-acd89(10)*acd89(71)
      acd89(70)=acd89(70)+acd89(71)
      acd89(70)=acd89(8)*acd89(70)
      acd89(71)=acd89(21)*acd89(22)
      acd89(72)=acd89(10)*acd89(74)
      acd89(71)=acd89(71)+acd89(72)
      acd89(71)=acd89(4)*acd89(71)
      acd89(72)=-acd89(21)*acd89(30)
      acd89(70)=acd89(71)+acd89(70)+acd89(44)+acd89(72)
      acd89(70)=acd89(5)*acd89(70)
      acd89(64)=-acd89(21)*acd89(64)
      acd89(67)=acd89(25)*acd89(67)
      acd89(71)=acd89(61)*acd89(62)
      acd89(72)=acd89(59)*acd89(60)
      acd89(73)=acd89(57)*acd89(58)
      acd89(74)=acd89(55)*acd89(56)
      acd89(75)=acd89(51)*acd89(52)
      acd89(76)=acd89(48)*acd89(49)
      acd89(77)=acd89(39)*acd89(40)
      acd89(78)=acd89(37)*acd89(38)
      acd89(79)=acd89(35)*acd89(54)
      acd89(80)=acd89(25)*acd89(34)
      acd89(80)=acd89(50)+acd89(80)
      acd89(80)=acd89(20)*acd89(80)
      acd89(81)=acd89(25)*acd89(32)
      acd89(81)=acd89(45)+acd89(81)
      acd89(81)=acd89(11)*acd89(81)
      acd89(82)=acd89(21)*acd89(31)
      acd89(82)=acd89(53)+acd89(82)
      acd89(82)=acd89(9)*acd89(82)
      acd89(83)=acd89(21)*acd89(29)
      acd89(83)=acd89(43)+acd89(83)
      acd89(83)=acd89(2)*acd89(83)
      acd89(63)=acd89(66)+acd89(63)+acd89(70)+acd89(69)+acd89(83)+acd89(82)+acd&
      &89(81)+acd89(80)+acd89(79)+acd89(78)+2.0_ki*acd89(77)+acd89(76)+acd89(75&
      &)+acd89(74)+acd89(73)+acd89(71)+acd89(72)+acd89(67)+acd89(64)
      brack(ninjaidxt1mu0)=acd89(68)
      brack(ninjaidxt0mu0)=acd89(63)
      brack(ninjaidxt0mu2)=acd89(65)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d89h0_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd89h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k2+k4
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
end module     p2_gg_httbar_d89h0l131_qp
