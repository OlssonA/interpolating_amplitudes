module     p0_ubaru_httbar_d4h9l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity9d4h9l131.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond_t, d => metric_tensor
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
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd4h9
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(13) :: acd4
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd4(1)=dotproduct(k2,ninjaE3)
      acd4(2)=dotproduct(ninjaE3,spvak1k2)
      acd4(3)=abb4(10)
      acd4(4)=dotproduct(ninjaE3,spval4l5)
      acd4(5)=abb4(12)
      acd4(6)=dotproduct(ninjaE3,spval4l3)
      acd4(7)=abb4(15)
      acd4(8)=dotproduct(ninjaE3,spval3k2)
      acd4(9)=abb4(17)
      acd4(10)=acd4(3)*acd4(1)
      acd4(11)=acd4(5)*acd4(4)
      acd4(12)=acd4(7)*acd4(6)
      acd4(13)=acd4(9)*acd4(8)
      acd4(10)=acd4(13)+acd4(12)+acd4(10)+acd4(11)
      acd4(10)=acd4(2)*acd4(10)
      brack(ninjaidxt2mu0)=acd4(10)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd4h9
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(47) :: acd4
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd4(1)=dotproduct(k2,ninjaE3)
      acd4(2)=dotproduct(ninjaE4,spvak1k2)
      acd4(3)=abb4(10)
      acd4(4)=dotproduct(k2,ninjaE4)
      acd4(5)=dotproduct(ninjaE3,spvak1k2)
      acd4(6)=dotproduct(ninjaE4,spval4l5)
      acd4(7)=abb4(12)
      acd4(8)=dotproduct(ninjaE4,spval3k2)
      acd4(9)=abb4(17)
      acd4(10)=dotproduct(ninjaE4,spval4l3)
      acd4(11)=abb4(15)
      acd4(12)=dotproduct(ninjaE3,spval4l5)
      acd4(13)=dotproduct(ninjaE3,spval3k2)
      acd4(14)=dotproduct(ninjaE3,spval4l3)
      acd4(15)=abb4(21)
      acd4(16)=dotproduct(k1,ninjaE3)
      acd4(17)=abb4(24)
      acd4(18)=dotproduct(k2,ninjaA)
      acd4(19)=dotproduct(ninjaA,spvak1k2)
      acd4(20)=abb4(13)
      acd4(21)=dotproduct(l4,ninjaE3)
      acd4(22)=dotproduct(ninjaA,ninjaE3)
      acd4(23)=dotproduct(ninjaA,spval4l5)
      acd4(24)=dotproduct(ninjaA,spval3k2)
      acd4(25)=dotproduct(ninjaA,spval4l3)
      acd4(26)=abb4(9)
      acd4(27)=abb4(28)
      acd4(28)=abb4(14)
      acd4(29)=abb4(31)
      acd4(30)=dotproduct(ninjaE3,spval4k2)
      acd4(31)=abb4(23)
      acd4(32)=dotproduct(k1,ninjaA)
      acd4(33)=dotproduct(l4,ninjaA)
      acd4(34)=dotproduct(ninjaA,ninjaA)
      acd4(35)=dotproduct(ninjaA,spval4k2)
      acd4(36)=abb4(11)
      acd4(37)=acd4(11)*acd4(10)
      acd4(38)=acd4(9)*acd4(8)
      acd4(39)=acd4(7)*acd4(6)
      acd4(40)=acd4(3)*acd4(4)
      acd4(37)=acd4(40)+acd4(37)+acd4(38)+acd4(39)
      acd4(37)=acd4(37)*acd4(5)
      acd4(38)=acd4(11)*acd4(14)
      acd4(39)=acd4(9)*acd4(13)
      acd4(40)=acd4(7)*acd4(12)
      acd4(41)=acd4(3)*acd4(1)
      acd4(38)=acd4(38)+acd4(39)+acd4(40)+acd4(41)
      acd4(39)=acd4(38)*acd4(2)
      acd4(37)=acd4(37)+acd4(39)+acd4(15)
      acd4(38)=acd4(19)*acd4(38)
      acd4(39)=acd4(11)*acd4(25)
      acd4(40)=acd4(9)*acd4(24)
      acd4(41)=acd4(7)*acd4(23)
      acd4(42)=acd4(3)*acd4(18)
      acd4(39)=acd4(41)+acd4(39)+acd4(40)+acd4(42)+acd4(26)
      acd4(40)=acd4(5)*acd4(39)
      acd4(41)=acd4(31)*acd4(30)
      acd4(42)=acd4(17)*acd4(16)
      acd4(43)=acd4(14)*acd4(29)
      acd4(44)=acd4(13)*acd4(28)
      acd4(45)=acd4(12)*acd4(27)
      acd4(46)=acd4(1)*acd4(20)
      acd4(47)=2.0_ki*acd4(22)+acd4(21)
      acd4(47)=acd4(15)*acd4(47)
      acd4(38)=acd4(40)+acd4(38)+acd4(47)+acd4(46)+acd4(45)+acd4(44)+acd4(43)+a&
      &cd4(41)+acd4(42)
      acd4(40)=ninjaP*acd4(37)
      acd4(39)=acd4(19)*acd4(39)
      acd4(41)=acd4(31)*acd4(35)
      acd4(42)=acd4(17)*acd4(32)
      acd4(43)=acd4(25)*acd4(29)
      acd4(44)=acd4(24)*acd4(28)
      acd4(45)=acd4(23)*acd4(27)
      acd4(46)=acd4(18)*acd4(20)
      acd4(47)=acd4(34)+acd4(33)
      acd4(47)=acd4(15)*acd4(47)
      acd4(39)=acd4(39)+acd4(47)+acd4(46)+acd4(45)+acd4(44)+acd4(43)+acd4(42)+a&
      &cd4(36)+acd4(41)+acd4(40)
      brack(ninjaidxt1mu0)=acd4(38)
      brack(ninjaidxt0mu0)=acd4(39)
      brack(ninjaidxt0mu2)=acd4(37)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d4h9_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd4h9
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k4+k5
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
end module     p0_ubaru_httbar_d4h9l131
