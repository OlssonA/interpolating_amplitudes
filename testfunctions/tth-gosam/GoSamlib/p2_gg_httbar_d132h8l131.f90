module     p2_gg_httbar_d132h8l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d132h8l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd132h8
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(24) :: acd132
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd132(1)=dotproduct(ninjaE3,spvak2e2)
      acd132(2)=dotproduct(ninjaE3,spvae2k2)
      acd132(3)=abb132(12)
      acd132(4)=dotproduct(ninjaE3,spvae2l5)
      acd132(5)=abb132(20)
      acd132(6)=dotproduct(ninjaE3,spvae2e1)
      acd132(7)=abb132(53)
      acd132(8)=dotproduct(ninjaE3,spvae2k1)
      acd132(9)=abb132(59)
      acd132(10)=dotproduct(ninjaE3,spvae2l4)
      acd132(11)=abb132(61)
      acd132(12)=dotproduct(ninjaE3,spvak1e2)
      acd132(13)=abb132(13)
      acd132(14)=dotproduct(ninjaE3,spvae1e2)
      acd132(15)=abb132(14)
      acd132(16)=dotproduct(ninjaE3,spval5e2)
      acd132(17)=abb132(28)
      acd132(18)=dotproduct(ninjaE3,spval4e2)
      acd132(19)=abb132(48)
      acd132(20)=acd132(5)*acd132(1)
      acd132(21)=acd132(13)*acd132(12)
      acd132(22)=acd132(15)*acd132(14)
      acd132(23)=acd132(17)*acd132(16)
      acd132(24)=acd132(19)*acd132(18)
      acd132(20)=acd132(24)+acd132(23)+acd132(22)+acd132(21)+acd132(20)
      acd132(20)=acd132(4)*acd132(20)
      acd132(21)=acd132(3)*acd132(2)
      acd132(22)=-acd132(7)*acd132(6)
      acd132(23)=acd132(9)*acd132(8)
      acd132(24)=-acd132(11)*acd132(10)
      acd132(21)=acd132(24)+acd132(23)+acd132(22)+acd132(21)
      acd132(21)=acd132(1)*acd132(21)
      acd132(20)=acd132(20)+acd132(21)
      brack(ninjaidxt2mu0)=acd132(20)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd132h8
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(65) :: acd132
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd132(1)=dotproduct(ninjaE3,spvak2e2)
      acd132(2)=dotproduct(ninjaE4,spvae2k2)
      acd132(3)=abb132(12)
      acd132(4)=dotproduct(ninjaE4,spvae2l5)
      acd132(5)=abb132(20)
      acd132(6)=dotproduct(ninjaE4,spvae2e1)
      acd132(7)=abb132(53)
      acd132(8)=dotproduct(ninjaE4,spvae2k1)
      acd132(9)=abb132(59)
      acd132(10)=dotproduct(ninjaE4,spvae2l4)
      acd132(11)=abb132(61)
      acd132(12)=dotproduct(ninjaE3,spvae2k2)
      acd132(13)=dotproduct(ninjaE4,spvak2e2)
      acd132(14)=dotproduct(ninjaE3,spvak1e2)
      acd132(15)=abb132(13)
      acd132(16)=dotproduct(ninjaE3,spvae2l5)
      acd132(17)=dotproduct(ninjaE4,spvak1e2)
      acd132(18)=dotproduct(ninjaE4,spvae1e2)
      acd132(19)=abb132(14)
      acd132(20)=dotproduct(ninjaE4,spval5e2)
      acd132(21)=abb132(28)
      acd132(22)=dotproduct(ninjaE4,spval4e2)
      acd132(23)=abb132(48)
      acd132(24)=dotproduct(ninjaE3,spvae1e2)
      acd132(25)=dotproduct(ninjaE3,spvae2e1)
      acd132(26)=dotproduct(ninjaE3,spvae2k1)
      acd132(27)=dotproduct(ninjaE3,spval5e2)
      acd132(28)=dotproduct(ninjaE3,spvae2l4)
      acd132(29)=dotproduct(ninjaE3,spval4e2)
      acd132(30)=dotproduct(ninjaA,spvak2e2)
      acd132(31)=dotproduct(ninjaA,spvae2k2)
      acd132(32)=dotproduct(ninjaA,spvak1e2)
      acd132(33)=dotproduct(ninjaA,spvae2l5)
      acd132(34)=dotproduct(ninjaA,spvae1e2)
      acd132(35)=dotproduct(ninjaA,spvae2e1)
      acd132(36)=dotproduct(ninjaA,spvae2k1)
      acd132(37)=dotproduct(ninjaA,spval5e2)
      acd132(38)=dotproduct(ninjaA,spvae2l4)
      acd132(39)=dotproduct(ninjaA,spval4e2)
      acd132(40)=abb132(17)
      acd132(41)=abb132(31)
      acd132(42)=abb132(18)
      acd132(43)=abb132(27)
      acd132(44)=abb132(22)
      acd132(45)=abb132(16)
      acd132(46)=abb132(19)
      acd132(47)=abb132(25)
      acd132(48)=abb132(36)
      acd132(49)=abb132(39)
      acd132(50)=abb132(15)
      acd132(51)=acd132(23)*acd132(22)
      acd132(52)=acd132(21)*acd132(20)
      acd132(53)=acd132(19)*acd132(18)
      acd132(54)=acd132(15)*acd132(17)
      acd132(55)=acd132(13)*acd132(5)
      acd132(51)=acd132(51)+acd132(53)+acd132(54)+acd132(52)+acd132(55)
      acd132(51)=acd132(51)*acd132(16)
      acd132(52)=acd132(11)*acd132(10)
      acd132(53)=acd132(9)*acd132(8)
      acd132(54)=acd132(7)*acd132(6)
      acd132(55)=acd132(3)*acd132(2)
      acd132(56)=acd132(4)*acd132(5)
      acd132(52)=-acd132(56)+acd132(52)-acd132(53)+acd132(54)-acd132(55)
      acd132(52)=acd132(52)*acd132(1)
      acd132(53)=acd132(11)*acd132(28)
      acd132(54)=acd132(9)*acd132(26)
      acd132(55)=acd132(7)*acd132(25)
      acd132(56)=acd132(3)*acd132(12)
      acd132(53)=-acd132(56)+acd132(55)+acd132(53)-acd132(54)
      acd132(54)=acd132(53)*acd132(13)
      acd132(55)=acd132(23)*acd132(29)
      acd132(56)=acd132(21)*acd132(27)
      acd132(57)=acd132(19)*acd132(24)
      acd132(58)=acd132(15)*acd132(14)
      acd132(55)=acd132(55)+acd132(56)+acd132(57)+acd132(58)
      acd132(56)=acd132(55)*acd132(4)
      acd132(51)=-acd132(54)+acd132(56)+acd132(51)-acd132(52)
      acd132(52)=acd132(11)*acd132(38)
      acd132(54)=acd132(9)*acd132(36)
      acd132(56)=acd132(7)*acd132(35)
      acd132(57)=acd132(3)*acd132(31)
      acd132(58)=acd132(33)*acd132(5)
      acd132(52)=-acd132(52)+acd132(54)-acd132(56)+acd132(57)+acd132(58)+acd132&
      &(40)
      acd132(54)=acd132(1)*acd132(52)
      acd132(55)=acd132(33)*acd132(55)
      acd132(53)=-acd132(30)*acd132(53)
      acd132(56)=acd132(23)*acd132(39)
      acd132(57)=acd132(21)*acd132(37)
      acd132(58)=acd132(19)*acd132(34)
      acd132(59)=acd132(15)*acd132(32)
      acd132(56)=acd132(56)+acd132(57)+acd132(58)+acd132(59)+acd132(43)
      acd132(57)=acd132(30)*acd132(5)
      acd132(57)=acd132(57)+acd132(56)
      acd132(57)=acd132(16)*acd132(57)
      acd132(58)=acd132(29)*acd132(49)
      acd132(59)=acd132(28)*acd132(48)
      acd132(60)=acd132(27)*acd132(47)
      acd132(61)=acd132(26)*acd132(46)
      acd132(62)=acd132(25)*acd132(45)
      acd132(63)=acd132(24)*acd132(44)
      acd132(64)=acd132(14)*acd132(42)
      acd132(65)=acd132(12)*acd132(41)
      acd132(53)=acd132(54)+acd132(57)+acd132(53)+acd132(55)+acd132(65)+acd132(&
      &64)+acd132(63)+acd132(62)+acd132(61)+acd132(60)+acd132(58)+acd132(59)
      acd132(54)=ninjaP*acd132(51)
      acd132(52)=acd132(30)*acd132(52)
      acd132(55)=acd132(33)*acd132(56)
      acd132(56)=acd132(39)*acd132(49)
      acd132(57)=acd132(38)*acd132(48)
      acd132(58)=acd132(37)*acd132(47)
      acd132(59)=acd132(36)*acd132(46)
      acd132(60)=acd132(35)*acd132(45)
      acd132(61)=acd132(34)*acd132(44)
      acd132(62)=acd132(32)*acd132(42)
      acd132(63)=acd132(31)*acd132(41)
      acd132(52)=acd132(54)+acd132(52)+acd132(55)+acd132(63)+acd132(62)+acd132(&
      &61)+acd132(60)+acd132(59)+acd132(58)+acd132(57)+acd132(50)+acd132(56)
      brack(ninjaidxt1mu0)=acd132(53)
      brack(ninjaidxt0mu0)=acd132(52)
      brack(ninjaidxt0mu2)=acd132(51)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d132h8_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd132h8
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k5
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
end module     p2_gg_httbar_d132h8l131
