module     p0_ubaru_httbar_d65h6l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity6d65h6l1.f90
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
      use p0_ubaru_httbar_abbrevd65h6
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc65(24)
      complex(ki) :: Qspk2
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspl5
      complex(ki) :: QspQ
      Qspk2 = dotproduct(Q,k2)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspl5 = dotproduct(Q,l5)
      QspQ = dotproduct(Q,Q)
      acc65(1)=abb65(8)
      acc65(2)=abb65(9)
      acc65(3)=abb65(10)
      acc65(4)=abb65(11)
      acc65(5)=abb65(15)
      acc65(6)=abb65(16)
      acc65(7)=abb65(18)
      acc65(8)=abb65(20)
      acc65(9)=abb65(29)
      acc65(10)=abb65(33)
      acc65(11)=abb65(36)
      acc65(12)=abb65(39)
      acc65(13)=abb65(42)
      acc65(14)=acc65(2)*Qspk2
      acc65(15)=acc65(3)*Qspval3k2
      acc65(14)=acc65(15)+acc65(1)+acc65(14)
      acc65(14)=Qspvak2k1*acc65(14)
      acc65(15)=acc65(6)*Qspval3k2
      acc65(16)=acc65(8)*Qspk2
      acc65(17)=Qspval5l4*acc65(10)
      acc65(18)=Qspval5l3*acc65(9)
      acc65(19)=Qspval5k1*acc65(5)
      acc65(20)=Qspval3k1*acc65(11)
      acc65(21)=Qspvak2l4*acc65(13)
      acc65(22)=Qspvak2l3*acc65(12)
      acc65(23)=Qspl5*acc65(7)
      acc65(24)=QspQ*acc65(4)
      brack=acc65(14)+acc65(15)+acc65(16)+acc65(17)+acc65(18)+acc65(19)+acc65(2&
      &0)+acc65(21)+acc65(22)+acc65(23)+acc65(24)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d65h6l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd65h6
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d65
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d65 = 0.0_ki
      d65 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d65, ki), aimag(d65), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d65h6l1
