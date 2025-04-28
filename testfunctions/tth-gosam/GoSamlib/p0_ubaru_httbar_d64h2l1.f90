module     p0_ubaru_httbar_d64h2l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity2d64h2l1.f90
   ! generator: buildfortran.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd64h2
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc64(26)
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      acc64(1)=abb64(9)
      acc64(2)=abb64(10)
      acc64(3)=abb64(11)
      acc64(4)=abb64(12)
      acc64(5)=abb64(13)
      acc64(6)=abb64(14)
      acc64(7)=abb64(16)
      acc64(8)=abb64(17)
      acc64(9)=abb64(18)
      acc64(10)=abb64(19)
      acc64(11)=abb64(20)
      acc64(12)=abb64(21)
      acc64(13)=abb64(30)
      acc64(14)=abb64(31)
      acc64(15)=-acc64(9)*Qspval5k2
      acc64(16)=-acc64(12)*Qspval5l3
      acc64(15)=acc64(16)+acc64(6)+acc64(15)
      acc64(15)=Qspvak2k1*acc64(15)
      acc64(16)=acc64(13)*Qspval5k2
      acc64(17)=acc64(14)*Qspval5l3
      acc64(18)=Qspval5k1*acc64(2)
      acc64(19)=Qspval4l5*acc64(10)
      acc64(20)=Qspval4k1*acc64(5)
      acc64(21)=Qspval3l5*acc64(3)
      acc64(22)=Qspval3k1*acc64(1)
      acc64(23)=Qspvak2l5*acc64(8)
      acc64(24)=Qspvak2l3*acc64(11)
      acc64(25)=Qspk2*acc64(4)
      acc64(26)=QspQ*acc64(7)
      brack=acc64(15)+acc64(16)+acc64(17)+acc64(18)+acc64(19)+acc64(20)+acc64(2&
      &1)+acc64(22)+acc64(23)+acc64(24)+acc64(25)+acc64(26)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d64h2l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd64h2
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d64
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d64 = 0.0_ki
      d64 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d64, ki), aimag(d64), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d64h2l1
