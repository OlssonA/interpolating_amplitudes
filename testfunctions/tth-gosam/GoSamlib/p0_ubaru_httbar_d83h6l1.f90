module     p0_ubaru_httbar_d83h6l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity6d83h6l1.f90
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
      use p0_ubaru_httbar_abbrevd83h6
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc83(25)
      complex(ki) :: Qspvak1k2
      complex(ki) :: QspQ
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspl5
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      QspQ = dotproduct(Q,Q)
      Qspk2 = dotproduct(Q,k2)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspl5 = dotproduct(Q,l5)
      acc83(1)=abb83(6)
      acc83(2)=abb83(7)
      acc83(3)=abb83(8)
      acc83(4)=abb83(11)
      acc83(5)=abb83(12)
      acc83(6)=abb83(13)
      acc83(7)=abb83(14)
      acc83(8)=abb83(16)
      acc83(9)=abb83(17)
      acc83(10)=abb83(22)
      acc83(11)=abb83(26)
      acc83(12)=abb83(27)
      acc83(13)=abb83(29)
      acc83(14)=abb83(30)
      acc83(15)=abb83(31)
      acc83(16)=acc83(5)*Qspvak1k2
      acc83(17)=QspQ*acc83(10)
      acc83(18)=Qspk2*acc83(1)
      acc83(16)=acc83(18)+acc83(17)+acc83(4)+acc83(16)
      acc83(16)=Qspvak2k1*acc83(16)
      acc83(17)=QspQ*acc83(6)
      acc83(18)=Qspk2*acc83(11)
      acc83(17)=acc83(18)+acc83(9)+acc83(17)
      acc83(17)=Qspk2*acc83(17)
      acc83(18)=-acc83(5)*Qspvak2l4
      acc83(18)=acc83(18)+acc83(15)
      acc83(18)=Qspval4k2*acc83(18)
      acc83(19)=-acc83(10)*Qspvak2l5
      acc83(19)=acc83(19)+acc83(14)
      acc83(19)=Qspval5k1*acc83(19)
      acc83(20)=acc83(13)*Qspval5l4
      acc83(21)=acc83(7)*Qspl5
      acc83(22)=Qspvak1k2*acc83(2)
      acc83(23)=Qspvak2l4*acc83(12)
      acc83(24)=Qspvak2l5*acc83(8)
      acc83(25)=QspQ*acc83(3)
      brack=acc83(16)+acc83(17)+acc83(18)+acc83(19)+acc83(20)+acc83(21)+acc83(2&
      &2)+acc83(23)+acc83(24)+acc83(25)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d83h6l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd83h6
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d83
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d83 = 0.0_ki
      d83 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d83, ki), aimag(d83), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d83h6l1
